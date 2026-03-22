from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from defer.baselines.policies import Policy
from defer.core.interfaces import (
    AgentAction,
    ContradictionSource,
    DelayedRevealEvent,
    EpisodeResult,
    EpisodeTrace,
    EpisodeTurn,
    Freshness,
    ProcedureGates,
    ToolCall,
    VerificationDecision,
    VerifierOutput,
)
from defer.core.correlated_verifier import (
    CorrelatedVerifier,
    CorrelatedVerifierConfig,
    load_failure_profiles,
)
from defer.core.verifier import UncertainVerifier, VerifierConfig
from defer.domains.contracts import default_contracts
from defer.domains.state import WorldState
from defer.domains.tools import TOOLS
from defer.domains.types import reset_tool_call_counter
from defer.sim.events import EventLoop
from defer.sim.scenario import Scenario
from defer.stress.faults import FaultInjector, FaultProfile
from defer.stress.perturb import perturb_prompt


_DEFAULT_PROFILES_PATH = Path(__file__).resolve().parent.parent / "configs" / "failure_profiles.json"


@dataclass(frozen=True)
class EnvironmentConfig:
    max_turns: int = 4
    turn_budget_multiplier: float = 1.5
    delayed_reveal_step_delta: int = 2
    stale_probability: float = 0.2
    provisional_probability: float = 0.25
    contradiction_probability: float = 0.15
    use_correlated_verifier: bool = False


def _scenario_hash_to_id(scenario: Scenario, seed: int) -> str:
    digest = hashlib.sha1(f"{scenario.scenario_id}:{seed}".encode("utf-8")).hexdigest()
    return digest[:12]


class SimulationEnvironment:
    def __init__(
        self,
        config: EnvironmentConfig | None = None,
    ) -> None:
        self.config = config or EnvironmentConfig()
        self.contracts = default_contracts()

    def run_episode(
        self,
        scenario: Scenario,
        policy: Policy,
        seed: int,
        epsilon: float,
        lambda_fault: float,
        repeat_index: int,
    ) -> EpisodeTrace:
        scenario_offset = int(hashlib.sha1(scenario.scenario_id.encode("utf-8")).hexdigest()[:8], 16) % 1_000_000
        # Keep stochastic environment realizations policy-invariant so paired
        # policy comparisons are not confounded by policy-specific RNG streams.
        episode_seed = seed + scenario_offset + repeat_index * 131
        reset_tool_call_counter(episode_seed)
        verifier_config = self._effective_verifier_config(
            scenario=scenario,
            epsilon=epsilon,
            lambda_fault=lambda_fault,
        )
        if self.config.use_correlated_verifier:
            correlated_config = self._build_correlated_config(verifier_config)
            verifier = CorrelatedVerifier(correlated_config, seed=episode_seed)
        else:
            verifier = UncertainVerifier(verifier_config, seed=episode_seed)
        fault_injector = FaultInjector(FaultProfile(lambda_fault=lambda_fault), seed=episode_seed + 7)
        event_loop = EventLoop()
        state = WorldState()
        turns: list[EpisodeTurn] = []
        delayed_events: list[DelayedRevealEvent] = []
        scenario_category = str(
            scenario.metadata.get("delayed_truth_category", scenario.delayed_truth_category or "A")
        )
        oracle_turns = 2 if scenario.has_delayed_truth else 1
        turn_budget = min(
            self.config.max_turns,
            max(1, int(math.ceil(oracle_turns * self.config.turn_budget_multiplier))),
        )

        last_decision: VerificationDecision | None = None
        last_freshness: str | None = None
        last_confidence = 1.0
        last_pending_reason: str | None = None
        unresolved_truth = bool(scenario.has_delayed_truth)
        fault_seen = False
        unsafe_retry_count = 0
        contradiction_observed = False
        contradiction_source: ContradictionSource | None = None
        success = False
        invalid_commit = False
        issued_primary_tool = False
        last_action: AgentAction | None = None

        perturbed_prompt = perturb_prompt(scenario.prompt, epsilon=epsilon, seed=episode_seed)

        for turn_idx in range(turn_budget):
            context = {
                "prompt": perturbed_prompt,
                "required_tool": scenario.required_tool,
                "tool_args": copy.deepcopy(scenario.tool_args),
                "last_verifier_decision": last_decision,
                "last_freshness": last_freshness,
                "last_confidence": last_confidence,
                "last_pending_reason": last_pending_reason,
                "unresolved_truth": unresolved_truth,
                "fault_seen": fault_seen,
                "has_delayed_truth": scenario.has_delayed_truth,
                "scenario_category": scenario_category,
                "delay_mechanism": scenario.delay_mechanism,
                "expects_irreversible": scenario.expects_irreversible,
                "requires_refresh": scenario.requires_refresh,
                "epsilon": epsilon,
                "lambda_fault": lambda_fault,
                "turn_idx": turn_idx,
                "turn_budget": turn_budget,
                "remaining_turns": max(0, turn_budget - turn_idx),
                "last_action": last_action,
                "domain": scenario.domain,
            }
            decision = policy.decide(context)

            turn = EpisodeTurn(
                turn_id=turn_idx,
                prompt=perturbed_prompt,
                selected_action=decision.action,
                unresolved_truth=unresolved_truth,
            )

            if decision.action in {AgentAction.DEFER_WAIT, AgentAction.DEFER_ASK_USER}:
                turn.refusal = decision.action == AgentAction.DEFER_ASK_USER
                emitted = event_loop.advance_to(turn_idx + 1)
                new_delays = self._to_delayed_reveals(emitted)
                delayed_events.extend(new_delays)
                if new_delays:
                    contradiction_event = next(
                        (event for event in new_delays if event.contradiction),
                        None,
                    )
                    if contradiction_event is not None:
                        contradiction_observed = True
                        contradiction_source = contradiction_event.contradiction_source
                        turn.contradiction_source = contradiction_source
                        turn.contradiction_observed = True
                        unresolved_truth = False
                    elif any(bool(event.revealed_truth.get("resolved", False)) for event in new_delays):
                        unresolved_truth = False
                if any(event.contradiction for event in delayed_events):
                    contradiction_observed = True
                    unresolved_truth = False
                turns.append(turn)
                last_action = turn.selected_action
                continue

            if decision.action in {AgentAction.DEFER_REFRESH, AgentAction.CROSS_CHECK_SECOND_TOOL}:
                tool_name = "refresh_state"
                tool_args = copy.deepcopy(decision.tool_args or {"keys": ["global"]})
            else:
                tool_name = decision.tool_name or scenario.required_tool
                tool_args = copy.deepcopy(decision.tool_args or scenario.tool_args)
                if decision.action == AgentAction.SAFE_COMMIT_REVERSIBLE and scenario.expects_irreversible:
                    tool_args["send_now"] = False
                if decision.action == AgentAction.FULL_COMMIT_IRREVERSIBLE and scenario.expects_irreversible:
                    tool_args["send_now"] = True

            turn.tool_call = ToolCall(tool_name=tool_name, args=copy.deepcopy(tool_args))

            if tool_name == scenario.required_tool:
                issued_primary_tool = True

            args_faulted, _, fault = fault_injector.inject(tool_name, tool_args)
            fault_seen = fault != "none"
            pre_state = state.flat()

            if args_faulted.get("__fault_timeout__") or args_faulted.get("__fault_rate_limit__"):
                unsafe_retry_count += 1
                turn.retries += 1
                turn.observation = {"fault": fault}
                turn.state_diff = {}
                last_decision = VerificationDecision.REJECT
                last_freshness = "stale"
                last_confidence = 0.2
                last_pending_reason = "external_state_may_change"
                turns.append(turn)
                last_action = turn.selected_action
                continue

            tool_fn = TOOLS.get(tool_name)
            if tool_fn is None:
                invalid_commit = True
                last_decision = VerificationDecision.REJECT
                last_freshness = "stale"
                last_confidence = 0.1
                turn.observation = {"error": "unknown_tool", "tool_name": tool_name, "fault": fault}
                turn.state_diff = {}
                turn.irreversible_commit = scenario.expects_irreversible
                turn.verifier_output = VerifierOutput(
                    decision=VerificationDecision.REJECT,
                    confidence=0.1,
                    freshness=Freshness.STALE,
                    pending_postconditions=[],
                    pending_postcondition_reason=None,
                    evidence_ids=[f"error:unknown_tool:{tool_name}"],
                )
                last_pending_reason = "unknown"
                turns.append(turn)
                last_action = turn.selected_action
                continue
            try:
                result = tool_fn(state, args_faulted)
            except KeyError:
                invalid_commit = True
                last_decision = VerificationDecision.REJECT
                last_freshness = "stale"
                last_confidence = 0.1
                turn.observation = {"error": "missing_required_argument", "fault": fault}
                turn.state_diff = {}
                turn.irreversible_commit = scenario.expects_irreversible
                turn.verifier_output = VerifierOutput(
                    decision=VerificationDecision.REJECT,
                    confidence=0.1,
                    freshness=Freshness.STALE,
                    pending_postconditions=[],
                    pending_postcondition_reason=None,
                    evidence_ids=["error:missing_required_argument"],
                )
                last_pending_reason = "unknown"
                turns.append(turn)
                last_action = turn.selected_action
                continue

            post_state = state.flat()
            contract = self.contracts[tool_name]
            verifier_output = verifier.verify(
                contract=contract,
                pre_state=pre_state,
                post_state=post_state,
                pending_fields=result.pending_fields if scenario.has_delayed_truth else [],
                fault_mode=fault,
                delayed_truth_category=scenario_category,
            )
            turn.verifier_output = verifier_output
            turn.observation = {**result.observation, "fault": fault}
            turn.state_diff = result.state_diff
            turn.used_stale_evidence = (
                verifier_output.freshness == Freshness.STALE
                and decision.action
                in {
                    AgentAction.SAFE_COMMIT_REVERSIBLE,
                    AgentAction.FULL_COMMIT_IRREVERSIBLE,
                }
            )
            turn.irreversible_commit = result.irreversible_commit

            last_decision = verifier_output.decision
            last_freshness = verifier_output.freshness.value
            last_confidence = verifier_output.confidence
            last_pending_reason = (
                verifier_output.pending_postcondition_reason.value
                if verifier_output.pending_postcondition_reason is not None
                else None
            )

            if verifier_output.decision == VerificationDecision.REJECT:
                invalid_commit = invalid_commit or result.irreversible_commit
            if verifier_output.decision == VerificationDecision.PROVISIONAL and scenario.has_delayed_truth:
                unresolved_truth = True
                event_id = f"reveal_{turn_idx}_{_scenario_hash_to_id(scenario, episode_seed)}"
                event_step = turn_idx + self._sample_delay_steps(
                    category=scenario_category,
                    episode_seed=episode_seed,
                    turn_idx=turn_idx,
                )

                def _callback_factory(local_turn_idx: int) -> Any:
                    def _callback() -> dict[str, Any]:
                        contradicted, source = verifier.maybe_contradict(
                            fault_mode=fault,
                            delayed_truth_category=scenario_category,
                        )
                        return {
                            "local_turn_idx": local_turn_idx,
                            "contradiction": contradicted,
                            "contradiction_source": source.value if source is not None else None,
                            "resolved": True,
                        }

                    return _callback

                event_loop.schedule(
                    step=event_step,
                    event_id=event_id,
                    callback=_callback_factory(turn_idx),
                    description=f"Delayed reveal for turn {turn_idx}",
                )
            elif verifier_output.decision == VerificationDecision.ACCEPT:
                unresolved_truth = False

            emitted = event_loop.advance_to(turn_idx + 1)
            delayed_emitted = self._to_delayed_reveals(emitted)
            if delayed_emitted:
                delayed_events.extend(delayed_emitted)
                contradiction_event = next(
                    (event for event in delayed_emitted if event.contradiction),
                    None,
                )
                if contradiction_event is not None:
                    contradiction_observed = True
                    contradiction_source = contradiction_event.contradiction_source
                    turn.contradiction_source = contradiction_source
                    unresolved_truth = False
                elif any(bool(event.revealed_truth.get("resolved", False)) for event in delayed_emitted):
                    unresolved_truth = False

            turn.contradiction_observed = contradiction_observed
            turns.append(turn)
            last_action = turn.selected_action

            if (
                issued_primary_tool
                and verifier_output.decision in {VerificationDecision.ACCEPT, VerificationDecision.PROVISIONAL}
                and not invalid_commit
                and not contradiction_observed
            ):
                success = True
                if decision.action not in {
                    AgentAction.CROSS_CHECK_SECOND_TOOL,
                    AgentAction.DEFER_REFRESH,
                }:
                    break

        # Drain all remaining delayed reveals so contradiction accounting is policy-invariant.
        pending = event_loop.drain_all()
        delayed_pending = self._to_delayed_reveals(pending)
        if delayed_pending:
            delayed_events.extend(delayed_pending)
            contradiction_observed = contradiction_observed or any(
                event.contradiction for event in delayed_pending
            )

        if contradiction_observed:
            success = False
        turn_budget_exhausted = not success and len(turns) >= turn_budget

        gates = ProcedureGates(
            policy_compliance=0 if invalid_commit else 1,
            policy_faithfulness=0 if contradiction_observed else 1,
            execution_consistency=0 if (unsafe_retry_count > 1 or turn_budget_exhausted) else 1,
            data_faithfulness=0 if contradiction_observed else 1,
            intent_adherence=1,
            question_fulfillment=1 if issued_primary_tool else 0,
        )
        # Corrupt success follows PAE semantics: terminal success with failed procedure gates.
        corrupt_success = bool(success and not gates.all_pass())
        result = EpisodeResult(
            success=success,
            invalid_commit=invalid_commit,
            corrupt_success=corrupt_success,
            unsafe_retry_count=unsafe_retry_count,
            turn_budget_exhausted=turn_budget_exhausted,
            explanation="Automated evaluation from deterministic simulator.",
            procedure_gates=gates,
        )
        episode_id = f"{scenario.scenario_id}_{policy.name}_{repeat_index}_{episode_seed}"
        return EpisodeTrace(
            episode_id=episode_id,
            scenario_id=scenario.scenario_id,
            domain=scenario.domain,
            delay_mechanism=scenario.delay_mechanism,
            policy_name=policy.name,
            seed=seed,
            epsilon=epsilon,
            lambda_fault=lambda_fault,
            repeat_index=repeat_index,
            scenario_category=scenario_category,
            turns=turns,
            delayed_events=delayed_events,
            final_state=state.flat(),
            result=result,
        )

    def _effective_verifier_config(
        self,
        scenario: Scenario,
        epsilon: float,
        lambda_fault: float,
    ) -> VerifierConfig:
        base_stale = self.config.stale_probability
        base_provisional = self.config.provisional_probability
        base_contradiction = (
            scenario.contradiction_probability_override
            if scenario.contradiction_probability_override is not None
            else self.config.contradiction_probability
        )
        delayed_bonus = 0.08 if scenario.has_delayed_truth else 0.0
        stale_probability = min(0.95, base_stale + 0.30 * lambda_fault + 0.10 * epsilon)
        provisional_probability = min(
            0.95,
            base_provisional + 0.22 * lambda_fault + 0.10 * epsilon + delayed_bonus,
        )
        contradiction_probability = min(
            0.95,
            base_contradiction + 0.35 * lambda_fault + 0.10 * epsilon + delayed_bonus,
        )
        return VerifierConfig(
            stale_probability=stale_probability,
            provisional_probability=provisional_probability,
            contradiction_probability=contradiction_probability,
            base_confidence=0.9,
            stale_penalty=0.25,
            provisional_penalty=0.2,
            reject_penalty=0.5,
        )

    def _build_correlated_config(
        self, base_config: VerifierConfig
    ) -> CorrelatedVerifierConfig:
        profiles_path = _DEFAULT_PROFILES_PATH
        profiles = load_failure_profiles(profiles_path) if profiles_path.exists() else {}
        return CorrelatedVerifierConfig(
            base_config=base_config,
            failure_profiles=profiles,
        )

    @staticmethod
    def _to_delayed_reveals(emitted: list[dict]) -> list[DelayedRevealEvent]:
        out: list[DelayedRevealEvent] = []
        for item in emitted:
            payload = item["payload"]
            out.append(
                DelayedRevealEvent(
                    event_id=item["event_id"],
                    time_step=item["time_step"],
                    description=item["description"],
                    revealed_truth=payload,
                    contradiction=bool(payload.get("contradiction", False)),
                    contradiction_source=(
                        ContradictionSource(payload["contradiction_source"])
                        if payload.get("contradiction_source")
                        else None
                    ),
                )
            )
        return out

    def _sample_delay_steps(self, category: str, episode_seed: int, turn_idx: int) -> int:
        """
        Approximate heavy-tailed reveal delays with category-dependent severity.
        """
        seed = episode_seed + turn_idx * 17 + ord(category[0]) * 31
        rng = random.Random(seed)
        if category == "A":
            mu, sigma = 0.20, 0.30
        elif category == "B":
            mu, sigma = 0.60, 0.55
        else:
            mu, sigma = 1.00, 0.75
        delay = int(max(1, round(rng.lognormvariate(mu, sigma))))
        return min(6, max(1, delay))
