from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from defer.core.interfaces import AgentAction, VerificationDecision


@dataclass
class PolicyDecision:
    action: AgentAction
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    reason: str = ""


class Policy(Protocol):
    name: str

    def decide(self, context: dict[str, Any]) -> PolicyDecision:
        ...


def _commit_action(expects_irreversible: bool) -> AgentAction:
    if expects_irreversible:
        return AgentAction.FULL_COMMIT_IRREVERSIBLE
    return AgentAction.SAFE_COMMIT_REVERSIBLE


def _stress_level(context: dict[str, Any]) -> float:
    epsilon = float(context.get("epsilon", 0.0))
    lambda_fault = float(context.get("lambda_fault", 0.0))
    return epsilon + lambda_fault


def _stress_corrupt_args(
    args: dict[str, Any],
    required_tool: str,
    *,
    epsilon: float,
    stress_level: float,
    threshold: float,
) -> dict[str, Any]:
    """
    Lightweight synthetic argument-grounding degradation under high stress.
    """
    if stress_level < threshold:
        return args
    out = dict(args)
    if required_tool == "send_email" and "subject" in out and epsilon >= 0.2:
        out["subject_line"] = out.pop("subject")
    elif required_tool == "create_calendar_event" and "start_time" in out and epsilon >= 0.3:
        out["starts_at"] = out.pop("start_time")
    elif required_tool == "upsert_sql_row" and "primary_key" in out and epsilon >= 0.3:
        out["pk"] = out.pop("primary_key")
    return out


class ReActPolicy:
    name = "react"

    def decide(self, context: dict[str, Any]) -> PolicyDecision:
        stress = _stress_level(context)
        epsilon = float(context.get("epsilon", 0.0))
        args = _stress_corrupt_args(
            context["tool_args"],
            context["required_tool"],
            epsilon=epsilon,
            stress_level=stress,
            threshold=0.35,
        )
        return PolicyDecision(
            action=_commit_action(bool(context.get("expects_irreversible", False))),
            tool_name=context["required_tool"],
            tool_args=args,
            reason="Directly act on the user request, with no uncertainty calibration.",
        )


class RuntimeVerifiedPolicy:
    name = "runtime_verification_only"

    def decide(self, context: dict[str, Any]) -> PolicyDecision:
        stress = _stress_level(context)
        epsilon = float(context.get("epsilon", 0.0))
        if context.get("last_verifier_decision") == VerificationDecision.REJECT:
            return PolicyDecision(
                action=AgentAction.DEFER_REFRESH,
                tool_name="refresh_state",
                tool_args={"keys": ["global"]},
                reason="Verifier rejected prior action; refresh and retry.",
            )
        if context.get("turn_idx", 0) == 0 and stress >= 0.55 and context.get("has_delayed_truth"):
            return PolicyDecision(
                action=AgentAction.CROSS_CHECK_SECOND_TOOL,
                tool_name="refresh_state",
                tool_args={"keys": ["global"]},
                reason="High stress prompts a one-time preflight refresh.",
            )
        args = _stress_corrupt_args(
            context["tool_args"],
            context["required_tool"],
            epsilon=epsilon,
            stress_level=stress,
            threshold=0.55,
        )
        return PolicyDecision(
            action=_commit_action(bool(context.get("expects_irreversible", False))),
            tool_name=context["required_tool"],
            tool_args=args,
            reason="Proceed because runtime verifier will gate output.",
        )


class CleanSFTPolicy:
    name = "clean_sft_only"

    def decide(self, context: dict[str, Any]) -> PolicyDecision:
        stress = _stress_level(context)
        epsilon = float(context.get("epsilon", 0.0))
        args = dict(context["tool_args"])
        if "tentative" in args:
            args["tentative"] = False
        args = _stress_corrupt_args(
            args,
            context["required_tool"],
            epsilon=epsilon,
            stress_level=stress,
            threshold=0.45,
        )
        return PolicyDecision(
            action=_commit_action(bool(context.get("expects_irreversible", False))),
            tool_name=context["required_tool"],
            tool_args=args,
            reason="Optimized for clean trajectories and direct completion.",
        )


class StressNoContractsPolicy:
    name = "stress_training_no_contracts"

    def decide(self, context: dict[str, Any]) -> PolicyDecision:
        stress = _stress_level(context)
        epsilon = float(context.get("epsilon", 0.0))
        args = _stress_corrupt_args(
            context["tool_args"],
            context["required_tool"],
            epsilon=epsilon,
            stress_level=stress,
            threshold=0.50,
        )
        if context.get("fault_seen"):
            return PolicyDecision(
                action=_commit_action(bool(context.get("expects_irreversible", False))),
                tool_name=context["required_tool"],
                tool_args=args,
                reason="Retry under stress without contract-aware caution.",
            )
        return PolicyDecision(
            action=_commit_action(bool(context.get("expects_irreversible", False))),
            tool_name=context["required_tool"],
            tool_args=args,
            reason="Proceed under stress training heuristics.",
        )


class PerfectVerifierPostTrainPolicy:
    name = "perfect_verifier_posttrain"

    def decide(self, context: dict[str, Any]) -> PolicyDecision:
        turn_idx = int(context.get("turn_idx", 0))
        if context.get("has_delayed_truth") and turn_idx == 0:
            return PolicyDecision(
                action=AgentAction.CROSS_CHECK_SECOND_TOOL,
                tool_name="refresh_state",
                tool_args={"keys": ["global"]},
                reason="Cross-check first with assumed perfect verifier.",
            )
        return PolicyDecision(
            action=_commit_action(bool(context.get("expects_irreversible", False))),
            tool_name=context["required_tool"],
            tool_args=context["tool_args"],
            reason="Act when checks are complete.",
        )


class DeferFullPolicy:
    name = "defer_full"

    def decide(self, context: dict[str, Any]) -> PolicyDecision:
        stress = _stress_level(context)
        freshness = context.get("last_freshness")
        confidence = context.get("last_confidence", 1.0)
        unresolved = context.get("unresolved_truth", False)
        irreversible = context.get("expects_irreversible", False)
        requires_refresh = context.get("requires_refresh", False)
        turn_idx = int(context.get("turn_idx", 0))
        last_action = context.get("last_action")
        pending_reason = context.get("last_pending_reason")
        remaining_turns = int(context.get("remaining_turns", 1))

        if unresolved and turn_idx == 0:
            if requires_refresh:
                return PolicyDecision(
                    action=AgentAction.CROSS_CHECK_SECOND_TOOL,
                    tool_name="refresh_state",
                    tool_args={"keys": ["global"]},
                    reason="Scenario requires a refresh before first commitment.",
                )
            return PolicyDecision(
                action=AgentAction.DEFER_WAIT,
                reason="Initial unresolved state: defer once before first commitment.",
            )

        if (
            unresolved
            and last_action == AgentAction.DEFER_WAIT
            and turn_idx == 1
            and remaining_turns > 1
            and context.get("scenario_category") in {"B", "C"}
        ):
            return PolicyDecision(
                action=AgentAction.CROSS_CHECK_SECOND_TOOL,
                tool_name="refresh_state",
                tool_args={"keys": ["global"]},
                reason="Still unresolved after initial wait; run one cross-check before commit.",
            )

        if unresolved and last_action == AgentAction.SAFE_COMMIT_REVERSIBLE:
            # Avoid repeated deferral loops: wait at most once, then proceed conservatively.
            if turn_idx <= 1 and remaining_turns > 1:
                return PolicyDecision(
                    action=AgentAction.DEFER_WAIT,
                    reason="Reversible action already taken; wait one turn for delayed reveal.",
                )
            return PolicyDecision(
                action=AgentAction.SAFE_COMMIT_REVERSIBLE,
                tool_name=context["required_tool"],
                tool_args=dict(context["tool_args"]),
                reason="Delayed reveal still unresolved; avoid over-deferral with conservative commit.",
            )

        if unresolved and irreversible:
            args = dict(context["tool_args"])
            args["send_now"] = False
            return PolicyDecision(
                action=AgentAction.SAFE_COMMIT_REVERSIBLE,
                tool_name=context["required_tool"],
                tool_args=args,
                reason="Use reversible commitment while truth is unresolved.",
            )
        threshold = 0.62 + 0.10 * stress
        if freshness == "stale" or confidence < threshold or requires_refresh:
            if remaining_turns <= 1:
                args = dict(context["tool_args"])
                if irreversible:
                    args["send_now"] = False
                return PolicyDecision(
                    action=AgentAction.SAFE_COMMIT_REVERSIBLE,
                    tool_name=context["required_tool"],
                    tool_args=args,
                    reason="Low turn budget under uncertainty; take conservative reversible action.",
                )
            if last_action in {AgentAction.CROSS_CHECK_SECOND_TOOL, AgentAction.DEFER_REFRESH} and turn_idx > 0:
                args = dict(context["tool_args"])
                if irreversible:
                    args["send_now"] = False
                return PolicyDecision(
                    action=AgentAction.SAFE_COMMIT_REVERSIBLE,
                    tool_name=context["required_tool"],
                    tool_args=args,
                    reason="Post-refresh execution to avoid deferral loops.",
                )
            # Refresh only once at episode start; otherwise favor conservative execution.
            if turn_idx == 0:
                return PolicyDecision(
                    action=AgentAction.DEFER_REFRESH,
                    tool_name="refresh_state",
                    tool_args={"keys": ["global"]},
                    reason="Initial uncertainty warrants one refresh before commit.",
                )
            args = dict(context["tool_args"])
            if irreversible:
                args["send_now"] = False
            return PolicyDecision(
                action=AgentAction.SAFE_COMMIT_REVERSIBLE,
                tool_name=context["required_tool"],
                tool_args=args,
                reason="Uncertainty persists after initial checks; commit conservatively.",
            )
        if pending_reason == "delayed_side_effect_not_observed" and remaining_turns > 1 and turn_idx == 0:
            return PolicyDecision(
                action=AgentAction.DEFER_WAIT,
                reason="Pending side effect not observed yet; wait one turn.",
            )
        if turn_idx == 0 and stress >= 0.55 and context.get("has_delayed_truth"):
            return PolicyDecision(
                action=AgentAction.CROSS_CHECK_SECOND_TOOL,
                tool_name="refresh_state",
                tool_args={"keys": ["global"]},
                reason="High stress + delayed truth triggers preflight cross-check.",
            )
        if irreversible:
            args = dict(context["tool_args"])
            # Commit irreversibly only when evidence is clearly resolved and confidence is high.
            if (not unresolved) and freshness == "fresh" and confidence >= max(0.88, threshold + 0.08):
                args["send_now"] = True
                return PolicyDecision(
                    action=AgentAction.FULL_COMMIT_IRREVERSIBLE,
                    tool_name=context["required_tool"],
                    tool_args=args,
                    reason="High-confidence resolved state permits irreversible commit.",
                )
            args["send_now"] = False
            return PolicyDecision(
                action=AgentAction.SAFE_COMMIT_REVERSIBLE,
                tool_name=context["required_tool"],
                tool_args=args,
                reason="Prefer reversible commitment under uncertainty.",
            )
        return PolicyDecision(
            action=AgentAction.SAFE_COMMIT_REVERSIBLE,
            tool_name=context["required_tool"],
            tool_args=context["tool_args"],
            reason="Commit when verifier evidence appears sufficient.",
        )


def policy_registry() -> dict[str, Policy]:
    policies: list[Policy] = [
        ReActPolicy(),
        RuntimeVerifiedPolicy(),
        CleanSFTPolicy(),
        StressNoContractsPolicy(),
        PerfectVerifierPostTrainPolicy(),
        DeferFullPolicy(),
    ]
    return {policy.name: policy for policy in policies}
