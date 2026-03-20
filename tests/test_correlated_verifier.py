from __future__ import annotations

from defer.core.correlated_verifier import (
    CorrelatedVerifier,
    CorrelatedVerifierConfig,
    FailureProfile,
    load_failure_profiles,
)
from defer.core.interfaces import ContractSpec, SideEffectType, VerificationDecision
from defer.core.verifier import VerifierConfig
from defer.domains.state import WorldState
from defer.sim.environment import EnvironmentConfig, SimulationEnvironment
from defer.sim.scenario import Scenario


def _simple_contract() -> ContractSpec:
    return ContractSpec(
        tool_name="create_calendar_event",
        preconditions=[{"field": "user_authorized", "op": "eq", "value": True}],
        postconditions=[{"field": "calendar_events", "op": "exists"}],
        side_effect_type=SideEffectType.REVERSIBLE,
        refresh_tools=["refresh_state"],
        failure_modes=["timeout"],
    )


def _make_verifier(
    profiles: dict[tuple[str, str], FailureProfile] | None = None,
    seed: int = 42,
) -> CorrelatedVerifier:
    config = CorrelatedVerifierConfig(
        base_config=VerifierConfig(),
        failure_profiles=profiles or {},
    )
    return CorrelatedVerifier(config, seed=seed)


def test_seed_determinism():
    v1 = _make_verifier(seed=99)
    v2 = _make_verifier(seed=99)
    contract = _simple_contract()
    state = WorldState()
    state.calendar_events["e1"] = state.calendar_events.get("e1", None) or type(
        "FakeEvent", (), {"model_dump": lambda self, **kw: {}}
    )()
    pre = {"user_authorized": True}
    post = {"user_authorized": True, "calendar_events": {"e1": {}}}

    results_1, results_2 = [], []
    for _ in range(20):
        r1 = v1.verify(contract, pre, post, pending_fields=["calendar_confirmed"], fault_mode="timeout", delayed_truth_category="B")
        r2 = v2.verify(contract, pre, post, pending_fields=["calendar_confirmed"], fault_mode="timeout", delayed_truth_category="B")
        results_1.append(r1.decision)
        results_2.append(r2.decision)
    assert results_1 == results_2


def test_consecutive_provisionals_increase_stale_rate():
    profiles = {("none", "A"): FailureProfile(stale_prob=0.1, provisional_prob=0.95, contradiction_prob=0.1)}
    v = _make_verifier(profiles=profiles, seed=42)
    contract = _simple_contract()
    pre = {"user_authorized": True}
    post = {"user_authorized": True, "calendar_events": {"e1": {}}}

    for _ in range(5):
        v.verify(contract, pre, post, pending_fields=["calendar_confirmed"], fault_mode="none", delayed_truth_category="A")

    assert v._consecutive_provisionals >= 4
    effective_stale = v._apply_consecutive_boost(0.1)
    assert effective_stale > 0.1


def test_profile_lookup_overrides_base():
    custom = FailureProfile(stale_prob=0.99, provisional_prob=0.99, contradiction_prob=0.99)
    profiles = {("schema_drift", "C"): custom}
    v = _make_verifier(profiles=profiles, seed=42)
    result = v._effective_probabilities("schema_drift", "C")
    assert result.stale_prob == 0.99
    assert result.provisional_prob == 0.99
    assert result.contradiction_prob == 0.99


def test_accept_resets_consecutive_counter():
    profiles = {("none", "A"): FailureProfile(stale_prob=0.0, provisional_prob=0.0, contradiction_prob=0.0)}
    v = _make_verifier(profiles=profiles, seed=42)
    v._consecutive_provisionals = 5
    contract = _simple_contract()
    pre = {"user_authorized": True}
    post = {"user_authorized": True, "calendar_events": {"e1": {}}}
    result = v.verify(contract, pre, post, fault_mode="none", delayed_truth_category="A")
    assert result.decision == VerificationDecision.ACCEPT
    assert v._consecutive_provisionals == 0


def test_missing_profile_falls_back_to_base():
    v = _make_verifier(profiles={}, seed=42)
    result = v._effective_probabilities("unknown_fault", "Z")
    assert result.stale_prob == VerifierConfig().stale_probability
    assert result.provisional_prob == VerifierConfig().provisional_probability
    assert result.contradiction_prob == VerifierConfig().contradiction_probability


def test_correlated_verifier_full_episode():
    env_config = EnvironmentConfig(use_correlated_verifier=True)
    env = SimulationEnvironment(config=env_config)
    scenario = Scenario(
        scenario_id="correlated_test_001",
        domain="calendar",
        prompt="Schedule a test meeting.",
        required_tool="create_calendar_event",
        tool_args={
            "title": "Test Meeting",
            "start_time": "2026-04-01T09:00",
            "end_time": "2026-04-01T10:00",
            "attendees": ["test@example.com"],
            "tentative": True,
        },
        expects_irreversible=False,
        has_delayed_truth=True,
        delayed_truth_category="B",
        delay_mechanism="eventual_consistency",
        metadata={"epsilon": 0.1, "lambda_fault": 0.1, "delayed_truth_category": "B"},
    )
    from defer.baselines.policies import DeferFullPolicy

    trace = env.run_episode(
        scenario=scenario,
        policy=DeferFullPolicy(),
        seed=42,
        epsilon=0.1,
        lambda_fault=0.1,
        repeat_index=0,
    )
    assert trace.episode_id
    assert len(trace.turns) > 0
