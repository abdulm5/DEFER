def test_comparisons_loaded_from_protocol():
    from scripts.evaluate_metrics import _load_protocol
    from pathlib import Path

    protocol = _load_protocol(Path("defer/configs/eval_protocol.yaml"))
    assert protocol["exists"]
    comparisons = protocol["payload"].get("comparisons", [])
    assert len(comparisons) >= 6
    assert ["defer_full", "clean_sft_only"] in comparisons


def test_missing_protocol_not_exists():
    from scripts.evaluate_metrics import _load_protocol
    from pathlib import Path

    protocol = _load_protocol(Path("/nonexistent/protocol.yaml"))
    assert not protocol["exists"]
