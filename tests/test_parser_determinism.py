from defer.core.contracts import parse_contract


def test_contract_parser_is_deterministic() -> None:
    raw = {
        "tool_name": "send_email",
        "preconditions": [{"field": "user_authorized", "op": "eq", "value": True}],
        "postconditions": [{"field": "emails", "op": "exists"}],
        "side_effect_type": "irreversible",
        "refresh_tools": ["refresh_state"],
        "failure_modes": ["timeout", "partial_response"],
    }
    a = parse_contract(raw).model_dump(mode="json")
    b = parse_contract(raw).model_dump(mode="json")
    assert a == b
