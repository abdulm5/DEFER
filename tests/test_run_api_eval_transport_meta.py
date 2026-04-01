from __future__ import annotations

import json
from pathlib import Path

from scripts.run_api_eval import run


class _DummyBaselinePolicy:
    name = "runtime_verification_only"

    def decide(self, context):  # pragma: no cover - not used in this test
        raise NotImplementedError


def _write_minimal_variants(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "variant_id": "v1",
        "task_id": "t1",
        "domain": "email",
        "split": "test",
        "prompt": "Send a status email",
        "required_tool": "send_email",
        "tool_args": {"subject": "Status", "body": "done", "to": ["a@example.com"]},
        "epsilon": 0.1,
        "lambda_fault": 0.05,
        "delay_setting": "immediate",
        "delayed_truth_category": "A",
        "delay_mechanism": "none",
        "fault_profile_id": "fp0",
        "expects_irreversible": False,
        "requires_refresh": False,
        "metadata": {},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_run_api_eval_writes_transport_metadata(monkeypatch, tmp_path: Path) -> None:
    variants = tmp_path / "variants.jsonl"
    output_dir = tmp_path / "api_eval"
    _write_minimal_variants(variants)
    monkeypatch.setenv("OPENAI_API_KEY", "token")
    monkeypatch.setattr(
        "scripts.run_api_eval.policy_registry",
        lambda: {"runtime_verification_only": _DummyBaselinePolicy()},
    )
    monkeypatch.setattr(
        "scripts.run_api_eval.run_policies",
        lambda scenarios, policies, config: ([], []),
    )

    run(
        variants=variants,
        output_dir=output_dir,
        split="test",
        repeats=1,
        seed=42,
        sampling_seed=1337,
        max_scenarios=1,
        api_policies=["frontier_model=gpt-4o"],
        include_baselines=["runtime_verification_only"],
        fallback_policy_name="runtime_verification_only",
        api_key_env="OPENAI_API_KEY",
        base_url="https://example.test/v1/chat/completions",
        auth_mode="api_key",
        api_key_header="api-key",
        query_params={"api-version": "2024-10-21"},
        max_new_tokens=64,
        temperature=0.0,
        top_p=1.0,
        timeout_seconds=15,
        max_retries=7,
        retry_backoff_seconds=0.5,
        retry_max_backoff_seconds=4.0,
    )

    meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))
    transport = meta["api_transport"]
    assert transport["auth_mode"] == "api_key"
    assert transport["api_key_header"] == "api-key"
    assert transport["query_params"] == {"api-version": "2024-10-21"}
    assert transport["sampling_seed"] == 1337
    assert transport["max_retries"] == 7
    assert transport["retry_backoff_seconds"] == 0.5
    assert transport["retry_max_backoff_seconds"] == 4.0


def test_run_api_eval_defaults_remain_openai_compatible(monkeypatch, tmp_path: Path) -> None:
    variants = tmp_path / "variants.jsonl"
    output_dir = tmp_path / "api_eval"
    _write_minimal_variants(variants)
    monkeypatch.setenv("OPENAI_API_KEY", "token")
    monkeypatch.setattr(
        "scripts.run_api_eval.policy_registry",
        lambda: {"runtime_verification_only": _DummyBaselinePolicy()},
    )
    monkeypatch.setattr(
        "scripts.run_api_eval.run_policies",
        lambda scenarios, policies, config: ([], []),
    )

    run(
        variants=variants,
        output_dir=output_dir,
        split="test",
        repeats=1,
        seed=42,
        max_scenarios=1,
        api_policies=["frontier_model=gpt-4o"],
        include_baselines=["runtime_verification_only"],
    )

    meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))
    transport = meta["api_transport"]
    assert transport["auth_mode"] == "bearer"
    assert transport["api_key_header"] == "api-key"
    assert transport["query_params"] == {}
