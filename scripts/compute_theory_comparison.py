from __future__ import annotations

import argparse
from pathlib import Path

from defer.analysis.theory import (
    DeferralCostModel,
    compare_empirical_to_optimal,
    multi_step_optimal_threshold,
    optimal_deferral_threshold,
)
from defer.core.interfaces import ReliabilityRecord
from defer.core.io import read_jsonl, write_json


def run(
    records_path: Path,
    output_dir: Path,
    policy_name: str,
) -> None:
    rows = read_jsonl(records_path)
    records = [ReliabilityRecord(**row) for row in rows if row.get("policy_name") == policy_name]

    model = DeferralCostModel()

    reversible_threshold = optimal_deferral_threshold(model, is_irreversible=False, remaining_turns=3)
    irreversible_threshold = optimal_deferral_threshold(model, is_irreversible=True, remaining_turns=3)

    multi_step_rev = multi_step_optimal_threshold(model, is_irreversible=False, max_turns=4)
    multi_step_irrev = multi_step_optimal_threshold(model, is_irreversible=True, max_turns=4)

    comparison = compare_empirical_to_optimal(records, model)

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "policy_name": policy_name,
        "cost_model": {
            "cost_premature_commit": model.cost_premature_commit,
            "cost_irreversible_error": model.cost_irreversible_error,
            "cost_deferral_per_turn": model.cost_deferral_per_turn,
            "cost_budget_exhaustion": model.cost_budget_exhaustion,
            "discount_factor": model.discount_factor,
        },
        "thresholds": {
            "reversible": reversible_threshold,
            "irreversible": irreversible_threshold,
        },
        "multi_step_thresholds": {
            "reversible": multi_step_rev,
            "irreversible": multi_step_irrev,
        },
        "empirical_comparison": comparison,
    }
    write_json(output_dir / "theory_comparison.json", result)
    print(f"Wrote theory comparison to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-name", type=str, default="defer_full")
    args = parser.parse_args()
    run(
        records_path=args.records_path,
        output_dir=args.output_dir,
        policy_name=args.policy_name,
    )


if __name__ == "__main__":
    main()
