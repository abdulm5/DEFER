from __future__ import annotations

from defer.analysis.failure_taxonomy import (
    REAL_WORLD_FAILURE_TAXONOMY,
    validate_taxonomy_coverage,
)
from defer.data.variants import DELAY_MECHANISMS


def test_every_delay_mechanism_has_taxonomy_entry():
    coverage = validate_taxonomy_coverage(DELAY_MECHANISMS)
    for mechanism, covered in coverage.items():
        assert covered, f"Missing taxonomy entry for delay mechanism: {mechanism}"


def test_taxonomy_entries_have_required_fields():
    required_fields = ["description", "real_world_examples", "mapped_delay_mechanism", "severity"]
    for mechanism, entry in REAL_WORLD_FAILURE_TAXONOMY.items():
        for field in required_fields:
            assert field in entry, f"Missing field '{field}' in taxonomy entry '{mechanism}'"
        assert len(entry["real_world_examples"]) >= 2, (
            f"Too few real-world examples for '{mechanism}': {len(entry['real_world_examples'])}"
        )
        assert entry["severity"] in {"critical", "high", "medium"}, (
            f"Invalid severity '{entry['severity']}' for '{mechanism}'"
        )
