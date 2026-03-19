from collections import Counter

from defer.data.seeds import generate_seed_tasks


def test_seed_split_ratio_per_domain() -> None:
    tasks = generate_seed_tasks(tasks_per_domain=300, seed=42)
    for domain in ["calendar", "email", "rest", "sql"]:
        counts = Counter(task.split for task in tasks if task.domain == domain)
        assert counts["train"] == 210
        assert counts["val"] == 30
        assert counts["test"] == 60
