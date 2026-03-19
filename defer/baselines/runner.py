from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from defer.baselines.policies import Policy
from defer.baselines.policies import policy_registry
from defer.core.interfaces import EpisodeTrace, ReliabilityRecord
from defer.metrics.reliability import trace_to_record
from defer.sim.environment import EnvironmentConfig, SimulationEnvironment
from defer.sim.scenario import Scenario


@dataclass(frozen=True)
class RunnerConfig:
    repeats: int = 5
    seed: int = 42


def run_policies(
    scenarios: Iterable[Scenario],
    policies: list[Policy],
    config: RunnerConfig | None = None,
) -> tuple[list[EpisodeTrace], list[ReliabilityRecord]]:
    cfg = config or RunnerConfig()
    env = SimulationEnvironment(
        EnvironmentConfig(
            max_turns=4,
            delayed_reveal_step_delta=2,
            stale_probability=0.2,
            provisional_probability=0.25,
            contradiction_probability=0.15,
        )
    )
    traces: list[EpisodeTrace] = []
    records: list[ReliabilityRecord] = []
    for policy in policies:
        for scenario in scenarios:
            for repeat_idx in range(cfg.repeats):
                trace = env.run_episode(
                    scenario=scenario,
                    policy=policy,
                    seed=cfg.seed,
                    epsilon=scenario.metadata.get("epsilon", 0.0),
                    lambda_fault=scenario.metadata.get("lambda_fault", 0.0),
                    repeat_index=repeat_idx,
                )
                traces.append(trace)
                records.append(trace_to_record(trace=trace, k=repeat_idx + 1))
    return traces, records


def run_baselines(
    scenarios: Iterable[Scenario],
    selected_policies: list[str] | None = None,
    config: RunnerConfig | None = None,
) -> tuple[list[EpisodeTrace], list[ReliabilityRecord]]:
    registry = policy_registry()
    policy_names = selected_policies or list(registry.keys())
    return run_policies(
        scenarios=scenarios,
        policies=[registry[name] for name in policy_names],
        config=config,
    )
