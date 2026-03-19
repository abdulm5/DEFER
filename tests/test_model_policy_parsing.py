from defer.baselines.model_policy import parse_policy_decision_text
from defer.core.interfaces import AgentAction


def _context() -> dict:
    return {
        "required_tool": "send_email",
        "tool_args": {"to": ["a@example.com"], "subject": "x", "body": "y", "send_now": True},
        "expects_irreversible": True,
    }


def test_parse_policy_decision_simple_json() -> None:
    text = '{"action":"FULL_COMMIT_IRREVERSIBLE","tool_name":"send_email","tool_args":{"subject":"A"}}'
    decision = parse_policy_decision_text(text=text, context=_context())
    assert decision is not None
    assert decision.action == AgentAction.FULL_COMMIT_IRREVERSIBLE
    assert decision.tool_name == "send_email"
    assert decision.tool_args["send_now"] is True


def test_parse_policy_decision_nested_tool_call() -> None:
    text = '{"selected_action":"SAFE_COMMIT_REVERSIBLE","tool_call":{"tool_name":"send_email","args":{"body":"ok"}}}'
    decision = parse_policy_decision_text(text=text, context=_context())
    assert decision is not None
    assert decision.action == AgentAction.SAFE_COMMIT_REVERSIBLE
    assert decision.tool_args["send_now"] is False
    assert decision.tool_name == "send_email"


def test_parse_policy_decision_alias_and_refresh_defaults() -> None:
    text = '{"action":"CROSS_CHECK","tool_args":{}}'
    decision = parse_policy_decision_text(text=text, context=_context())
    assert decision is not None
    assert decision.action == AgentAction.CROSS_CHECK_SECOND_TOOL
    assert decision.tool_name == "refresh_state"
    assert decision.tool_args == {"keys": ["global"]}


def test_parse_policy_decision_invalid_returns_none() -> None:
    decision = parse_policy_decision_text(text="not-json-response", context=_context())
    assert decision is None
