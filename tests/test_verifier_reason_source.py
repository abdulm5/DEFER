from defer.core.interfaces import ContractSpec, ContradictionSource, PendingPostconditionReason
from defer.core.verifier import UncertainVerifier, VerifierConfig


def test_verifier_emits_provisional_reason() -> None:
    contract = ContractSpec(
        tool_name="upsert_api_resource",
        preconditions=[{"field": "user_authorized", "op": "eq", "value": True}],
        postconditions=[{"field": "api_resources", "op": "exists"}],
        side_effect_type="reversible",
        refresh_tools=["refresh_state"],
        failure_modes=["partial_response"],
    )
    verifier = UncertainVerifier(
        VerifierConfig(stale_probability=0.0, provisional_probability=1.0),
        seed=1,
    )
    out = verifier.verify(
        contract=contract,
        pre_state={"user_authorized": True},
        post_state={"user_authorized": True, "api_resources": {"r1": {"ok": True}}},
        pending_fields=[],
        fault_mode="partial_response",
    )
    assert out.pending_postcondition_reason == PendingPostconditionReason.PARTIAL_OUTPUT


def test_contradiction_source_is_typed() -> None:
    verifier = UncertainVerifier(VerifierConfig(contradiction_probability=1.0), seed=1)
    contradicted, source = verifier.maybe_contradict(
        fault_mode="schema_drift",
        delayed_truth_category="C",
    )
    assert contradicted is True
    assert source == ContradictionSource.SCHEMA_CONFLICT
