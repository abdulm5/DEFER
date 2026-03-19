from defer.data.seeds import generate_seed_tasks
from defer.data.variants import generate_variants


def test_delay_mechanism_assignment_matches_delay_setting() -> None:
    seeds = generate_seed_tasks(tasks_per_domain=4, seed=42)
    variants = generate_variants(seeds, seed=42)
    assert variants
    immediate = [v for v in variants if v.delay_setting == "immediate"]
    delayed = [v for v in variants if v.delay_setting == "delayed"]
    assert all(v.delay_mechanism == "none" for v in immediate)
    assert all(v.delay_mechanism != "none" for v in delayed)
