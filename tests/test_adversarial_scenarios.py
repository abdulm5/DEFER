from __future__ import annotations

from defer.baselines.policies import DeferFullPolicy
from defer.sim.adversarial_scenarios import (
    AdversarialScenarioConfig,
    _CATEGORY_SPECS,
    generate_adversarial_scenarios,
)
from defer.sim.environment import EnvironmentConfig, SimulationEnvironment


def test_generates_requested_count():
    config = AdversarialScenarioConfig(n_scenarios=60, seed=42)
    scenarios = generate_adversarial_scenarios(config)
    assert len(scenarios) == 60


def test_all_categories_appear():
    config = AdversarialScenarioConfig(n_scenarios=200, seed=42)
    scenarios = generate_adversarial_scenarios(config)
    categories = {s.metadata.get("adversarial_category") for s in scenarios}
    assert categories == set(_CATEGORY_SPECS)


def test_each_scenario_produces_valid_trace():
    config = AdversarialScenarioConfig(n_scenarios=12, seed=42)
    scenarios = generate_adversarial_scenarios(config)
    env = SimulationEnvironment(EnvironmentConfig())
    policy = DeferFullPolicy()
    for scenario in scenarios[:6]:
        trace = env.run_episode(
            scenario=scenario,
            policy=policy,
            seed=42,
            epsilon=scenario.metadata.get("epsilon", 0.0),
            lambda_fault=scenario.metadata.get("lambda_fault", 0.0),
            repeat_index=0,
        )
        assert trace.episode_id
        assert len(trace.turns) > 0


def test_adversarial_category_in_metadata():
    config = AdversarialScenarioConfig(n_scenarios=30, seed=42)
    scenarios = generate_adversarial_scenarios(config)
    for s in scenarios:
        assert "adversarial_category" in s.metadata
        assert s.metadata["adversarial_category"] in _CATEGORY_SPECS


def test_at_least_one_category_challenges_defer_full():
    config = AdversarialScenarioConfig(n_scenarios=60, seed=42)
    scenarios = generate_adversarial_scenarios(config)
    env = SimulationEnvironment(EnvironmentConfig())
    policy = DeferFullPolicy()

    by_category: dict[str, list[bool]] = {}
    for scenario in scenarios:
        trace = env.run_episode(
            scenario=scenario,
            policy=policy,
            seed=42,
            epsilon=scenario.metadata.get("epsilon", 0.0),
            lambda_fault=scenario.metadata.get("lambda_fault", 0.0),
            repeat_index=0,
        )
        cat = scenario.metadata.get("adversarial_category", "unknown")
        by_category.setdefault(cat, []).append(trace.result.success)

    failure_rates = {}
    for cat, results in by_category.items():
        failure_rates[cat] = 1.0 - (sum(results) / len(results))

    assert any(rate > 0.1 for rate in failure_rates.values()), (
        f"No adversarial category showed meaningful failures: {failure_rates}"
    )
