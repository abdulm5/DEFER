from __future__ import annotations

import csv
import json
from pathlib import Path

from defer.core.io import write_jsonl


def test_run_seed_sweep_writes_merged_fallback_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from scripts import run_seed_sweep as module

    variants_path = tmp_path / "variants.jsonl"
    output_dir = tmp_path / "sweep"
    seeds_config = tmp_path / "seeds.json"

    write_jsonl(variants_path, [{"variant_id": "v1"}])
    seeds_config.write_text(
        json.dumps({"primary_model_seeds": [1], "confirmatory_model_seeds": [2]}),
        encoding="utf-8",
    )

    def fake_run_baselines_run(
        variants,
        output_dir,
        split,
        repeats,
        seed,
        max_scenarios,
        domains=None,
        include_delay_mechanisms=None,
        exclude_delay_mechanisms=None,
    ):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "reliability_records.jsonl", [{"policy_name": "defer_full"}])
        write_jsonl(out / "episode_traces.jsonl", [{"policy_name": "defer_full", "turns": []}])
        with (out / "fallback_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "policy",
                    "total_decisions",
                    "parse_failures",
                    "fallback_calls",
                    "fallback_rate",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "policy": "defer_full",
                    "total_decisions": 10,
                    "parse_failures": seed,
                    "fallback_calls": seed,
                    "fallback_rate": seed / 10.0,
                }
            )

    monkeypatch.setattr(module, "run_baselines_run", fake_run_baselines_run)

    module.run(
        variants=variants_path,
        output_dir=output_dir,
        seeds_config=seeds_config,
        repeats=1,
        split="test",
        max_scenarios=1,
    )

    merged_fallback = output_dir / "fallback_metrics.csv"
    assert merged_fallback.exists()
    rows = list(csv.DictReader(merged_fallback.open("r", encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert row["policy"] == "defer_full"
    assert int(row["total_decisions"]) == 20
    assert int(row["parse_failures"]) == 3
    assert int(row["fallback_calls"]) == 3
    assert abs(float(row["fallback_rate"]) - 0.15) < 1e-9
