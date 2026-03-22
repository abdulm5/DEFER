import math


def test_bh_known_values():
    from scripts.evaluate_metrics import _benjamini_hochberg
    raw = [0.01, 0.04, 0.03, 0.20, 0.50]
    adj = _benjamini_hochberg(raw)
    assert len(adj) == 5
    assert adj[0] <= 0.051  # 0.01 * 5/1 = 0.05
    assert all(a <= 1.0 for a in adj)


def test_bh_single():
    from scripts.evaluate_metrics import _benjamini_hochberg
    assert _benjamini_hochberg([0.03]) == [0.03]


def test_bh_all_ones():
    from scripts.evaluate_metrics import _benjamini_hochberg
    assert _benjamini_hochberg([1.0, 1.0, 1.0]) == [1.0, 1.0, 1.0]


def test_bh_handles_nan():
    from scripts.evaluate_metrics import _benjamini_hochberg
    result = _benjamini_hochberg([0.01, float("nan"), 0.05])
    assert not math.isnan(result[0])
    assert math.isnan(result[1])
    assert not math.isnan(result[2])
