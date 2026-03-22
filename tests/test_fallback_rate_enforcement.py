import pytest


def test_high_fallback_rate_raises():
    from scripts.run_checkpoint_eval import _validate_fallback_rates
    rows = [{"policy": "test_pol", "fallback_rate": 0.15}]
    with pytest.raises(ValueError, match="exceeds"):
        _validate_fallback_rates(rows, max_rate=0.10)


def test_acceptable_fallback_rate_passes():
    from scripts.run_checkpoint_eval import _validate_fallback_rates
    rows = [{"policy": "test_pol", "fallback_rate": 0.05}]
    _validate_fallback_rates(rows, max_rate=0.10)
