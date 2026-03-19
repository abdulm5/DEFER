from __future__ import annotations

from defer.core.interfaces import ContractSpec, SideEffectType


def default_contracts() -> dict[str, ContractSpec]:
    return {
        "create_calendar_event": ContractSpec(
            tool_name="create_calendar_event",
            preconditions=[
                {"field": "user_authorized", "op": "eq", "value": True},
            ],
            postconditions=[
                {"field": "calendar_events", "op": "exists"},
            ],
            side_effect_type=SideEffectType.REVERSIBLE,
            refresh_tools=["refresh_state"],
            failure_modes=["timeout", "partial_response", "schema_drift", "rate_limit"],
        ),
        "send_email": ContractSpec(
            tool_name="send_email",
            preconditions=[{"field": "user_authorized", "op": "eq", "value": True}],
            postconditions=[{"field": "emails", "op": "exists"}],
            side_effect_type=SideEffectType.IRREVERSIBLE,
            refresh_tools=["refresh_state"],
            failure_modes=["timeout", "rate_limit", "partial_response", "schema_drift"],
        ),
        "upsert_api_resource": ContractSpec(
            tool_name="upsert_api_resource",
            preconditions=[{"field": "user_authorized", "op": "eq", "value": True}],
            postconditions=[{"field": "api_resources", "op": "exists"}],
            side_effect_type=SideEffectType.REVERSIBLE,
            refresh_tools=["refresh_state"],
            failure_modes=["timeout", "schema_drift", "missing_field", "partial_response"],
        ),
        "upsert_sql_row": ContractSpec(
            tool_name="upsert_sql_row",
            preconditions=[{"field": "user_authorized", "op": "eq", "value": True}],
            postconditions=[{"field": "sql_rows", "op": "exists"}],
            side_effect_type=SideEffectType.REVERSIBLE,
            refresh_tools=["refresh_state"],
            failure_modes=["timeout", "schema_drift", "missing_field", "partial_response"],
        ),
        "refresh_state": ContractSpec(
            tool_name="refresh_state",
            preconditions=[],
            postconditions=[],
            side_effect_type=SideEffectType.REVERSIBLE,
            refresh_tools=[],
            failure_modes=[],
        ),
    }
