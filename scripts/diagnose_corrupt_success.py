from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from defer.core.interfaces import EpisodeTrace
from defer.core.io import read_jsonl, write_jsonl


def run(
    traces_path: Path,
    output_dir: Path,
    policy: str,
    categories: set[str] | None,
    max_samples: int,
) -> None:
    traces = [EpisodeTrace(**row) for row in read_jsonl(traces_path)]
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = [trace for trace in traces if trace.policy_name == policy]
    if categories is not None:
        traces = [trace for trace in traces if trace.scenario_category in categories]

    if not traces:
        pd.DataFrame(
            columns=["scenario_category", "episodes", "task_success", "gated_success", "corrupt_success", "ier"]
        ).to_csv(output_dir / "summary_by_category.csv", index=False)
        pd.DataFrame(columns=["scenario_category", "contradiction_source", "count"]).to_csv(
            output_dir / "contradiction_sources.csv",
            index=False,
        )
        pd.DataFrame(columns=["scenario_category", "action_pattern", "count"]).to_csv(
            output_dir / "action_patterns.csv",
            index=False,
        )
        write_jsonl(output_dir / "sample_corrupt_traces.jsonl", [])
        print(f"No matching traces for policy={policy}.")
        return

    by_category: dict[str, list[EpisodeTrace]] = defaultdict(list)
    for trace in traces:
        by_category[trace.scenario_category].append(trace)

    summary_rows: list[dict] = []
    contradiction_rows: list[dict] = []
    pattern_rows: list[dict] = []
    sample_rows: list[dict] = []

    for category, cat_traces in sorted(by_category.items()):
        episodes = len(cat_traces)
        task_success = sum(1 for trace in cat_traces if trace.result.success) / max(1, episodes)
        gated_success = sum(
            1
            for trace in cat_traces
            if trace.result.success and trace.result.procedure_gates.all_pass()
        ) / max(1, episodes)
        corrupt_success = sum(1 for trace in cat_traces if trace.result.corrupt_success) / max(1, episodes)
        ier = sum(1 for trace in cat_traces if trace.result.invalid_commit) / max(1, episodes)
        summary_rows.append(
            {
                "scenario_category": category,
                "episodes": episodes,
                "task_success": task_success,
                "gated_success": gated_success,
                "corrupt_success": corrupt_success,
                "ier": ier,
            }
        )

        contradiction_counter: Counter[str] = Counter()
        pattern_counter: Counter[str] = Counter()
        corrupt_traces = [trace for trace in cat_traces if trace.result.corrupt_success]
        for trace in corrupt_traces:
            source = "none"
            for event in trace.delayed_events:
                if event.contradiction:
                    source = (
                        event.contradiction_source.value
                        if event.contradiction_source is not None
                        else "unknown"
                    )
                    break
            contradiction_counter[source] += 1
            pattern = " > ".join(turn.selected_action.value for turn in trace.turns)
            pattern_counter[pattern] += 1

        contradiction_rows.extend(
            {
                "scenario_category": category,
                "contradiction_source": source,
                "count": count,
            }
            for source, count in contradiction_counter.most_common()
        )
        pattern_rows.extend(
            {
                "scenario_category": category,
                "action_pattern": pattern,
                "count": count,
            }
            for pattern, count in pattern_counter.most_common(20)
        )

        for trace in corrupt_traces[:max_samples]:
            sample_rows.append(
                {
                    "episode_id": trace.episode_id,
                    "scenario_id": trace.scenario_id,
                    "scenario_category": trace.scenario_category,
                    "domain": trace.domain,
                    "policy_name": trace.policy_name,
                    "epsilon": trace.epsilon,
                    "lambda_fault": trace.lambda_fault,
                    "delay_mechanism": trace.delay_mechanism,
                    "turns": [turn.model_dump(mode="json") for turn in trace.turns],
                    "delayed_events": [event.model_dump(mode="json") for event in trace.delayed_events],
                    "result": trace.result.model_dump(mode="json"),
                }
            )

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_by_category.csv", index=False)
    pd.DataFrame(contradiction_rows).to_csv(output_dir / "contradiction_sources.csv", index=False)
    pd.DataFrame(pattern_rows).to_csv(output_dir / "action_patterns.csv", index=False)
    write_jsonl(output_dir / "sample_corrupt_traces.jsonl", sample_rows)
    print(f"Wrote corrupt-success diagnostics under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traces-path",
        type=Path,
        default=Path("artifacts/runs/episode_traces.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/diagnostics/corrupt_success"),
    )
    parser.add_argument("--policy", type=str, default="defer_full")
    parser.add_argument(
        "--categories",
        type=str,
        default="B,C",
        help="Comma-separated scenario categories to include. Empty string keeps all.",
    )
    parser.add_argument("--max-samples", type=int, default=15)
    args = parser.parse_args()
    categories = {item.strip() for item in args.categories.split(",") if item.strip()} or None
    run(
        traces_path=args.traces_path,
        output_dir=args.output_dir,
        policy=args.policy,
        categories=categories,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
