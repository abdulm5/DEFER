from __future__ import annotations

import hashlib
import random

from defer.data.schema import SeedTask, VariantTask
from defer.stress.perturb import perturb_prompt


EPSILONS = [0.0, 0.1, 0.2, 0.3]
FAULT_LEVELS = [0.0, 0.1, 0.2, 0.3]
DELAY_SETTINGS = ["immediate", "delayed"]
DELAY_MECHANISMS = [
    "eventual_consistency",
    "async_job_completion",
    "concurrent_edit_conflict",
    "stale_schema_cache",
    "delayed_authorization",
    "cross_tool_evidence_lag",
]


def _delayed_truth_category(epsilon: float, lambda_fault: float, delay_setting: str) -> str:
    if delay_setting == "immediate":
        return "A"
    score = epsilon + lambda_fault
    if score <= 0.25:
        return "A"
    if score <= 0.45:
        return "B"
    return "C"


def _delay_mechanism(task_id: str, epsilon: float, lambda_fault: float, delay_setting: str) -> str:
    if delay_setting == "immediate":
        return "none"
    digest = hashlib.sha1(f"{task_id}:{epsilon}:{lambda_fault}".encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(DELAY_MECHANISMS)
    return DELAY_MECHANISMS[idx]


def generate_variants(seed_tasks: list[SeedTask], seed: int) -> list[VariantTask]:
    rng = random.Random(seed)
    variants: list[VariantTask] = []
    for task in seed_tasks:
        for epsilon in EPSILONS:
            for lambda_fault in FAULT_LEVELS:
                for delay_setting in DELAY_SETTINGS:
                    delayed_truth_category = _delayed_truth_category(
                        epsilon=epsilon,
                        lambda_fault=lambda_fault,
                        delay_setting=delay_setting,
                    )
                    delay_mechanism = _delay_mechanism(
                        task_id=task.task_id,
                        epsilon=epsilon,
                        lambda_fault=lambda_fault,
                        delay_setting=delay_setting,
                    )
                    stable_hash = int(hashlib.sha1(task.task_id.encode("utf-8")).hexdigest()[:8], 16)
                    prompt = perturb_prompt(task.prompt, epsilon=epsilon, seed=seed + stable_hash % 10_000)
                    # Inject distractor text for higher epsilon.
                    if epsilon >= 0.2 and rng.random() < 0.5:
                        prompt = f"{prompt} Context: ignore unrelated chatter from prior thread."
                    fault_profile_id = f"fault_{lambda_fault:.1f}"
                    variant_id = (
                        f"{task.task_id}_e{str(epsilon).replace('.', '')}"
                        f"_l{str(lambda_fault).replace('.', '')}_{delay_setting}"
                    )
                    variants.append(
                        VariantTask(
                            variant_id=variant_id,
                            task_id=task.task_id,
                            domain=task.domain,
                            split=task.split,
                            prompt=prompt,
                            required_tool=task.required_tool,
                            tool_args=task.tool_args,
                            epsilon=epsilon,
                            lambda_fault=lambda_fault,
                            delay_setting=delay_setting,
                            delayed_truth_category=delayed_truth_category,
                            delay_mechanism=delay_mechanism,
                            fault_profile_id=fault_profile_id,
                            expects_irreversible=task.expects_irreversible,
                            requires_refresh=task.requires_refresh,
                            metadata={
                                "template_id": task.template_id,
                                "delayed_truth": delay_setting == "delayed",
                                "delayed_truth_category": delayed_truth_category,
                                "delay_mechanism": delay_mechanism,
                            },
                        )
                    )
    return variants


def as_json_rows(variants: list[VariantTask]) -> list[dict]:
    return [variant.model_dump(mode="json") for variant in variants]
