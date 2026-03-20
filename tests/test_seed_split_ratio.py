from collections import Counter

from defer.data.seeds import DOMAIN_TEMPLATES, generate_seed_tasks


def test_seed_split_ratio_per_domain() -> None:
    tasks = generate_seed_tasks(tasks_per_domain=300, seed=42)
    for domain in DOMAIN_TEMPLATES:
        counts = Counter(task.split for task in tasks if task.domain == domain)
        total = sum(counts.values())
        assert total == 300, f"{domain}: expected 300 tasks, got {total}"
        train_ratio = counts["train"] / total
        val_ratio = counts["val"] / total
        test_ratio = counts["test"] / total
        assert 0.60 <= train_ratio <= 0.80, f"{domain}: train ratio {train_ratio:.2f} outside [0.60, 0.80]"
        assert 0.05 <= val_ratio <= 0.15, f"{domain}: val ratio {val_ratio:.2f} outside [0.05, 0.15]"
        assert 0.15 <= test_ratio <= 0.25, f"{domain}: test ratio {test_ratio:.2f} outside [0.15, 0.25]"
