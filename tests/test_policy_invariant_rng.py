from __future__ import annotations

from defer.baselines.policies import PolicyDecision
from defer.core.interfaces import AgentAction
from defer.sim.environment import SimulationEnvironment
from defer.sim.scenario import Scenario


class _TwoStepPolicy:
    def __init__(self, name: str) -> None:
        self.name = name

    def decide(self, context: dict) -> PolicyDecision:
        if int(context.get("turn_idx", 0)) == 0:
            return PolicyDecision(
                action=AgentAction.CROSS_CHECK_SECOND_TOOL,
                tool_name="refresh_state",
                tool_args={"keys": ["global"]},
            )
        return PolicyDecision(
            action=AgentAction.SAFE_COMMIT_REVERSIBLE,
            tool_name=context["required_tool"],
            tool_args=dict(context["tool_args"]),
        )


def test_policy_name_does_not_change_prompt_or_fault_stream() -> None:
    scenario = Scenario(
        scenario_id="rng_invariance_001",
        domain="calendar",
        prompt=(
            "Schedule an urgent email notification to grant file upload permission "
            "and deploy webhook confirm database status."
        ),
        required_tool="create_calendar_event",
        tool_args={
            "title": "Weekly Sync",
            "start_time": "2026-04-01T09:00",
            "end_time": "2026-04-01T10:00",
            "attendees": ["alex@example.com"],
            "tentative": True,
        },
        expects_irreversible=False,
        requires_refresh=False,
        has_delayed_truth=True,
        delayed_truth_category="B",
        delay_mechanism="eventual_consistency",
        metadata={"epsilon": 0.3, "lambda_fault": 0.3},
    )
    env = SimulationEnvironment()

    trace_a = env.run_episode(
        scenario=scenario,
        policy=_TwoStepPolicy(name="policy_a"),
        seed=42,
        epsilon=0.3,
        lambda_fault=0.3,
        repeat_index=0,
    )
    trace_b = env.run_episode(
        scenario=scenario,
        policy=_TwoStepPolicy(name="policy_b"),
        seed=42,
        epsilon=0.3,
        lambda_fault=0.3,
        repeat_index=0,
    )

    assert trace_a.turns[0].prompt == trace_b.turns[0].prompt
    faults_a = [turn.observation.get("fault") for turn in trace_a.turns]
    faults_b = [turn.observation.get("fault") for turn in trace_b.turns]
    assert faults_a == faults_b
