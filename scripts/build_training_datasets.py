from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from defer.core.interfaces import EpisodeTrace
from defer.core.io import read_jsonl, write_json, write_jsonl
from defer.training.formatting import render_pair_response, render_trace_response


def _split_tag(key: str, seed: int, val_ratio: float) -> str:
    digest = hashlib.sha1(f"{key}:{seed}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def _allow_scenario(
    scenario_id: str,
    scenario_meta: dict[str, dict[str, str]],
    include_domains: set[str] | None,
    exclude_domains: set[str] | None,
    include_delay_mechanisms: set[str] | None,
    exclude_delay_mechanisms: set[str] | None,
) -> bool:
    meta = scenario_meta.get(scenario_id, {})
    domain = meta.get("domain", "unknown")
    mechanism = meta.get("delay_mechanism", "none")
    if include_domains is not None and domain not in include_domains:
        return False
    if exclude_domains is not None and domain in exclude_domains:
        return False
    if include_delay_mechanisms is not None and mechanism not in include_delay_mechanisms:
        return False
    if exclude_delay_mechanisms is not None and mechanism in exclude_delay_mechanisms:
        return False
    return True


def _build_scenario_meta(rows: list[EpisodeTrace]) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    for trace in rows:
        meta[trace.scenario_id] = {"domain": trace.domain, "delay_mechanism": trace.delay_mechanism}
    return meta


def _build_sft_rows(
    traces: list[EpisodeTrace],
    scenario_meta: dict[str, dict[str, str]],
    seed: int,
    val_ratio: float,
    include_domains: set[str] | None,
    exclude_domains: set[str] | None,
    include_delay_mechanisms: set[str] | None,
    exclude_delay_mechanisms: set[str] | None,
) -> tuple[list[dict], list[dict]]:
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for trace in traces:
        if not _allow_scenario(
            scenario_id=trace.scenario_id,
            scenario_meta=scenario_meta,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_delay_mechanisms=include_delay_mechanisms,
            exclude_delay_mechanisms=exclude_delay_mechanisms,
        ):
            continue
        if not (
            trace.result.success
            and trace.result.procedure_gates.all_pass()
            and not trace.result.corrupt_success
            and not trace.result.turn_budget_exhausted
        ):
            continue
        prompt = trace.turns[0].prompt if trace.turns else ""
        row = {
            "scenario_id": trace.scenario_id,
            "policy_name": trace.policy_name,
            "prompt": prompt,
            "response": render_trace_response(trace),
        }
        split = _split_tag(key=trace.scenario_id, seed=seed, val_ratio=val_ratio)
        if split == "val":
            val_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, val_rows


def _build_dpo_rows(
    pairs_path: Path,
    scenario_meta: dict[str, dict[str, str]],
    seed: int,
    val_ratio: float,
    include_domains: set[str] | None,
    exclude_domains: set[str] | None,
    include_delay_mechanisms: set[str] | None,
    exclude_delay_mechanisms: set[str] | None,
) -> tuple[list[dict], list[dict]]:
    rows = read_jsonl(pairs_path)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for row in rows:
        if not _allow_scenario(
            scenario_id=row["scenario_id"],
            scenario_meta=scenario_meta,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_delay_mechanisms=include_delay_mechanisms,
            exclude_delay_mechanisms=exclude_delay_mechanisms,
        ):
            continue
        out = {
            "scenario_id": row["scenario_id"],
            "pair_type": row.get("pair_type", "general_commit_timing"),
            "pair_polarity": row.get("pair_polarity", "unknown"),
            "quality_margin": row.get("quality_margin", 0.0),
            "prompt": row.get("prompt", ""),
            "chosen": render_pair_response(row.get("chosen", [])),
            "rejected": render_pair_response(row.get("rejected", [])),
        }
        split = _split_tag(key=row["scenario_id"], seed=seed + 9_973, val_ratio=val_ratio)
        if split == "val":
            val_rows.append(out)
        else:
            train_rows.append(out)
    return train_rows, val_rows


def _build_perfect_verifier_dpo_rows(
    pairs_path: Path,
    scenario_meta: dict[str, dict[str, str]],
    seed: int,
    val_ratio: float,
    include_domains: set[str] | None,
    exclude_domains: set[str] | None,
    include_delay_mechanisms: set[str] | None,
    exclude_delay_mechanisms: set[str] | None,
) -> tuple[list[dict], list[dict]]:
    rows = read_jsonl(pairs_path)
    filtered: list[dict] = []
    for row in rows:
        if not _allow_scenario(
            scenario_id=row["scenario_id"],
            scenario_meta=scenario_meta,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_delay_mechanisms=include_delay_mechanisms,
            exclude_delay_mechanisms=exclude_delay_mechanisms,
        ):
            continue
        if not str(row.get("scenario_id", "")).endswith("_immediate"):
            continue
        filtered.append(
            {
                "scenario_id": row["scenario_id"],
                "pair_type": "perfect_verifier_immediate_truth",
                "pair_polarity": row.get("pair_polarity", "unknown"),
                "quality_margin": row.get("quality_margin", 0.0),
                "prompt": row.get("prompt", ""),
                "chosen": render_pair_response(row.get("chosen", [])),
                "rejected": render_pair_response(row.get("rejected", [])),
            }
        )
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for row in filtered:
        split = _split_tag(key=row["scenario_id"], seed=seed + 29_993, val_ratio=val_ratio)
        if split == "val":
            val_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, val_rows


def run(
    traces_path: Path,
    pairs_path: Path,
    output_dir: Path,
    seed: int,
    val_ratio: float,
    include_domains: set[str] | None = None,
    exclude_domains: set[str] | None = None,
    include_delay_mechanisms: set[str] | None = None,
    exclude_delay_mechanisms: set[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = [EpisodeTrace(**row) for row in read_jsonl(traces_path)]
    scenario_meta = _build_scenario_meta(traces)

    sft_train, sft_val = _build_sft_rows(
        traces=traces,
        scenario_meta=scenario_meta,
        seed=seed,
        val_ratio=val_ratio,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        include_delay_mechanisms=include_delay_mechanisms,
        exclude_delay_mechanisms=exclude_delay_mechanisms,
    )
    dpo_train, dpo_val = _build_dpo_rows(
        pairs_path=pairs_path,
        scenario_meta=scenario_meta,
        seed=seed,
        val_ratio=val_ratio,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        include_delay_mechanisms=include_delay_mechanisms,
        exclude_delay_mechanisms=exclude_delay_mechanisms,
    )
    perfect_train, perfect_val = _build_perfect_verifier_dpo_rows(
        pairs_path=pairs_path,
        scenario_meta=scenario_meta,
        seed=seed,
        val_ratio=val_ratio,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        include_delay_mechanisms=include_delay_mechanisms,
        exclude_delay_mechanisms=exclude_delay_mechanisms,
    )
    if not sft_val and len(sft_train) > 1:
        sft_val.append(sft_train.pop(0))
    if not dpo_val and len(dpo_train) > 1:
        dpo_val.append(dpo_train.pop(0))
    if (not perfect_train and not perfect_val) and dpo_train:
        perfect_train = [dict(row, pair_type="perfect_verifier_proxy") for row in dpo_train]
        perfect_val = [dict(row, pair_type="perfect_verifier_proxy") for row in dpo_val]
    if not perfect_val and len(perfect_train) > 1:
        perfect_val.append(perfect_train.pop(0))

    sft_train_path = output_dir / "sft_train.jsonl"
    sft_val_path = output_dir / "sft_val.jsonl"
    dpo_train_path = output_dir / "dpo_train.jsonl"
    dpo_val_path = output_dir / "dpo_val.jsonl"
    perfect_train_path = output_dir / "dpo_perfect_train.jsonl"
    perfect_val_path = output_dir / "dpo_perfect_val.jsonl"

    write_jsonl(sft_train_path, sft_train)
    write_jsonl(sft_val_path, sft_val)
    write_jsonl(dpo_train_path, dpo_train)
    write_jsonl(dpo_val_path, dpo_val)
    write_jsonl(perfect_train_path, perfect_train)
    write_jsonl(perfect_val_path, perfect_val)

    meta = {
        "seed": seed,
        "val_ratio": val_ratio,
        "include_domains": sorted(include_domains) if include_domains is not None else "all",
        "exclude_domains": sorted(exclude_domains) if exclude_domains is not None else [],
        "include_delay_mechanisms": (
            sorted(include_delay_mechanisms) if include_delay_mechanisms is not None else "all"
        ),
        "exclude_delay_mechanisms": (
            sorted(exclude_delay_mechanisms) if exclude_delay_mechanisms is not None else []
        ),
        "sft_train_rows": len(sft_train),
        "sft_val_rows": len(sft_val),
        "dpo_train_rows": len(dpo_train),
        "dpo_val_rows": len(dpo_val),
        "dpo_perfect_train_rows": len(perfect_train),
        "dpo_perfect_val_rows": len(perfect_val),
        "paths": {
            "sft_train": str(sft_train_path),
            "sft_val": str(sft_val_path),
            "dpo_train": str(dpo_train_path),
            "dpo_val": str(dpo_val_path),
            "dpo_perfect_train": str(perfect_train_path),
            "dpo_perfect_val": str(perfect_val_path),
        },
    }
    write_json(output_dir / "training_data_meta.json", meta)
    print(f"Wrote training datasets under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traces-path",
        type=Path,
        default=Path("artifacts/paper_run_v1/baseline_runs/episode_traces.jsonl"),
    )
    parser.add_argument(
        "--pairs-path",
        type=Path,
        default=Path("artifacts/paper_run_v1/data/dpo_pairs.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/paper_run_v1/data/training"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--include-domains", type=str, default="")
    parser.add_argument("--exclude-domains", type=str, default="")
    parser.add_argument("--include-delay-mechanisms", type=str, default="")
    parser.add_argument("--exclude-delay-mechanisms", type=str, default="")
    args = parser.parse_args()
    include_domains = {d.strip() for d in args.include_domains.split(",") if d.strip()} or None
    exclude_domains = {d.strip() for d in args.exclude_domains.split(",") if d.strip()} or None
    include_delay_mechanisms = (
        {m.strip() for m in args.include_delay_mechanisms.split(",") if m.strip()} or None
    )
    exclude_delay_mechanisms = (
        {m.strip() for m in args.exclude_delay_mechanisms.split(",") if m.strip()} or None
    )
    run(
        traces_path=args.traces_path,
        pairs_path=args.pairs_path,
        output_dir=args.output_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        include_delay_mechanisms=include_delay_mechanisms,
        exclude_delay_mechanisms=exclude_delay_mechanisms,
    )


if __name__ == "__main__":
    main()
