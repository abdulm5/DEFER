from __future__ import annotations

import pandas as pd
import pytest

from scripts.evaluate_metrics import _assert_protocol_pairwise_completeness


def test_pairwise_completeness_raises_when_expected_rows_missing() -> None:
    protocol = {
        "exists": True,
        "payload": {"comparisons": [["defer_full", "runtime_verification_only"]]},
    }
    pairwise = pd.DataFrame(
        [
            {
                "metric": "AURS",
                "policy_a": "defer_full",
                "policy_b": "runtime_verification_only",
                "diff_point": 0.1,
            }
        ]
    )
    with pytest.raises(ValueError, match="Protocol pairwise completeness check failed"):
        _assert_protocol_pairwise_completeness(
            pairwise_df=pairwise,
            protocol=protocol,
            required_metrics=("AURS", "DCS"),
        )


def test_pairwise_completeness_passes_when_all_expected_rows_present() -> None:
    protocol = {
        "exists": True,
        "payload": {"comparisons": [["defer_full", "runtime_verification_only"]]},
    }
    pairwise = pd.DataFrame(
        [
            {"metric": "AURS", "policy_a": "defer_full", "policy_b": "runtime_verification_only"},
            {"metric": "DCS", "policy_a": "defer_full", "policy_b": "runtime_verification_only"},
        ]
    )
    _assert_protocol_pairwise_completeness(
        pairwise_df=pairwise,
        protocol=protocol,
        required_metrics=("AURS", "DCS"),
    )
