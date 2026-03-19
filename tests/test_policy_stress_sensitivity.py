from defer.baselines.policies import DeferFullPolicy, ReActPolicy
from defer.core.interfaces import AgentAction


def test_react_policy_corrupts_args_under_high_stress() -> None:
    policy = ReActPolicy()
    base_context = {
        "required_tool": "send_email",
        "tool_args": {
            "subject": "Status update",
            "body": "Ping",
            "to": ["alex@example.com"],
            "send_now": True,
        },
    }
    low = policy.decide({**base_context, "epsilon": 0.0, "lambda_fault": 0.0})
    high = policy.decide({**base_context, "epsilon": 0.3, "lambda_fault": 0.5})

    assert low.tool_args is not None and "subject" in low.tool_args
    assert high.tool_args is not None and "subject" not in high.tool_args
    assert "subject_line" in high.tool_args


def test_defer_policy_prefers_reversible_action_when_unresolved_irreversible() -> None:
    policy = DeferFullPolicy()
    decision_turn0 = policy.decide(
        {
            "required_tool": "send_email",
            "tool_args": {"subject": "x", "body": "y", "to": ["a@b.com"], "send_now": True},
            "epsilon": 0.2,
            "lambda_fault": 0.2,
            "unresolved_truth": True,
            "expects_irreversible": True,
            "requires_refresh": False,
            "last_freshness": "fresh",
            "last_confidence": 0.9,
            "turn_idx": 0,
            "last_action": None,
            "has_delayed_truth": True,
        }
    )
    assert decision_turn0.action == AgentAction.DEFER_WAIT

    decision_turn1 = policy.decide(
        {
            "required_tool": "send_email",
            "tool_args": {"subject": "x", "body": "y", "to": ["a@b.com"], "send_now": True},
            "epsilon": 0.2,
            "lambda_fault": 0.2,
            "unresolved_truth": True,
            "expects_irreversible": True,
            "requires_refresh": False,
            "last_freshness": "fresh",
            "last_confidence": 0.9,
            "turn_idx": 1,
            "last_action": AgentAction.DEFER_WAIT,
            "has_delayed_truth": True,
            "remaining_turns": 2,
        }
    )
    assert decision_turn1.action == AgentAction.SAFE_COMMIT_REVERSIBLE
    assert decision_turn1.tool_args is not None and decision_turn1.tool_args.get("send_now") is False
