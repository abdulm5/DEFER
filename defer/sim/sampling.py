from __future__ import annotations

import hashlib

from defer.sim.scenario import Scenario


def _scenario_digest(scenario: Scenario, seed: int, salt: str) -> str:
    epsilon = scenario.metadata.get("epsilon", "")
    lambda_fault = scenario.metadata.get("lambda_fault", "")
    token = (
        f"{salt}|{seed}|{scenario.scenario_id}|{scenario.domain}|"
        f"{scenario.delay_mechanism}|{epsilon}|{lambda_fault}"
    )
    return hashlib.sha1(token.encode("utf-8")).hexdigest()


def _cell_key(scenario: Scenario) -> tuple[str, float, float]:
    epsilon = float(scenario.metadata.get("epsilon", 0.0))
    lambda_fault = float(scenario.metadata.get("lambda_fault", 0.0))
    return (scenario.domain, epsilon, lambda_fault)


def deterministic_sample_scenarios(
    scenarios: list[Scenario],
    max_scenarios: int,
    seed: int,
    *,
    salt: str = "",
) -> list[Scenario]:
    """
    Deterministically sample scenarios without relying on source row order.

    This avoids head-of-file truncation bias when `max_scenarios` is smaller
    than the available scenario count.
    """
    if max_scenarios <= 0:
        return []
    if len(scenarios) <= max_scenarios:
        return list(scenarios)

    grouped: dict[tuple[str, float, float], list[Scenario]] = {}
    for scenario in scenarios:
        grouped.setdefault(_cell_key(scenario), []).append(scenario)

    # Deterministic pseudo-random order within each stratification cell.
    for key in grouped:
        grouped[key] = sorted(
            grouped[key],
            key=lambda scenario: (_scenario_digest(scenario, seed=seed, salt=salt), scenario.scenario_id),
        )

    # Deterministic pseudo-random order over cells.
    cell_order = sorted(
        grouped.keys(),
        key=lambda key: hashlib.sha1(
            f"{salt}|{seed}|cell|{key[0]}|{key[1]}|{key[2]}".encode("utf-8")
        ).hexdigest(),
    )

    selected: list[Scenario] = []
    cursor: dict[tuple[str, float, float], int] = {key: 0 for key in cell_order}
    while len(selected) < max_scenarios:
        progressed = False
        for key in cell_order:
            idx = cursor[key]
            bucket = grouped[key]
            if idx >= len(bucket):
                continue
            selected.append(bucket[idx])
            cursor[key] = idx + 1
            progressed = True
            if len(selected) >= max_scenarios:
                break
        if not progressed:
            break
    return selected
