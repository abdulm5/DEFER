from __future__ import annotations

from typing import Any

from defer.core.interfaces import ConditionSpec, ContractSpec


def parse_contract(raw: dict[str, Any]) -> ContractSpec:
    """
    Deterministic contract parser used by tests and runtime.
    """
    normalized = {
        "tool_name": raw["tool_name"],
        "preconditions": [ConditionSpec(**c).model_dump() for c in raw.get("preconditions", [])],
        "postconditions": [ConditionSpec(**c).model_dump() for c in raw.get("postconditions", [])],
        "side_effect_type": raw["side_effect_type"],
        "refresh_tools": list(raw.get("refresh_tools", [])),
        "failure_modes": list(raw.get("failure_modes", [])),
    }
    return ContractSpec(**normalized)


def _match_condition(state: dict[str, Any], condition: ConditionSpec) -> bool:
    exists = condition.field in state
    current = state.get(condition.field)
    op = condition.op
    target = condition.value

    if op == "exists":
        return exists
    if not exists:
        return False
    if op == "eq":
        return current == target
    if op == "ne":
        return current != target
    if op == "gt":
        return current > target
    if op == "gte":
        return current >= target
    if op == "lt":
        return current < target
    if op == "lte":
        return current <= target
    if op == "in":
        return current in (target or [])
    raise ValueError(f"Unsupported condition op: {op}")


def check_preconditions(contract: ContractSpec, state: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for condition in contract.preconditions:
        if not _match_condition(state, condition):
            failures.append(f"precondition_failed:{condition.field}:{condition.op}")
    return failures


def check_postconditions(contract: ContractSpec, state: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for condition in contract.postconditions:
        if not _match_condition(state, condition):
            failures.append(f"postcondition_failed:{condition.field}:{condition.op}")
    return failures

