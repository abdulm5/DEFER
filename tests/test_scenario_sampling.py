from __future__ import annotations

from defer.sim.sampling import deterministic_sample_scenarios
from defer.sim.scenario import Scenario


def _scenario(idx: int, domain: str) -> Scenario:
    return Scenario(
        scenario_id=f"s_{idx:04d}",
        domain=domain,
        prompt=f"Prompt {idx}",
        required_tool="refresh_state",
        tool_args={"keys": ["global"]},
        metadata={"epsilon": 0.1, "lambda_fault": 0.1},
    )


def test_deterministic_sampling_breaks_head_bias() -> None:
    scenarios = [_scenario(i, "calendar") for i in range(100)] + [
        _scenario(i + 100, "email") for i in range(100)
    ]
    sampled = deterministic_sample_scenarios(scenarios, max_scenarios=50, seed=42, salt="test")
    sampled_domains = {s.domain for s in sampled}
    assert "calendar" in sampled_domains
    assert "email" in sampled_domains


def test_deterministic_sampling_reproducible_and_seeded() -> None:
    scenarios = [_scenario(i, "calendar") for i in range(120)]
    sample_a = deterministic_sample_scenarios(scenarios, max_scenarios=40, seed=7, salt="test")
    sample_b = deterministic_sample_scenarios(scenarios, max_scenarios=40, seed=7, salt="test")
    sample_c = deterministic_sample_scenarios(scenarios, max_scenarios=40, seed=8, salt="test")
    ids_a = [s.scenario_id for s in sample_a]
    ids_b = [s.scenario_id for s in sample_b]
    ids_c = [s.scenario_id for s in sample_c]
    assert ids_a == ids_b
    assert ids_a != ids_c
