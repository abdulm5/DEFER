from defer.core.interfaces import ContractSpec
from defer.core.verifier import UncertainVerifier


def test_verifier_is_seed_deterministic() -> None:
    contract = ContractSpec(
        tool_name="upsert_api_resource",
        preconditions=[{"field": "user_authorized", "op": "eq", "value": True}],
        postconditions=[{"field": "api_resources", "op": "exists"}],
        side_effect_type="reversible",
        refresh_tools=["refresh_state"],
        failure_modes=["schema_drift"],
    )
    pre_state = {"user_authorized": True}
    post_state = {"user_authorized": True, "api_resources": {"a": {"x": 1}}}

    verifier_a = UncertainVerifier(seed=123)
    verifier_b = UncertainVerifier(seed=123)
    out_a = verifier_a.verify(contract, pre_state, post_state, pending_fields=["api_replication"])
    out_b = verifier_b.verify(contract, pre_state, post_state, pending_fields=["api_replication"])
    assert out_a.model_dump(mode="json") == out_b.model_dump(mode="json")
