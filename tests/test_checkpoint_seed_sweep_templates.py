import pytest

from scripts.run_checkpoint_seed_sweep import _format_policy_templates


def test_format_policy_templates_substitutes_seed() -> None:
    specs = _format_policy_templates(
        templates=[
            "defer_full=/tmp/defer_full_seed_{seed}/model",
            "clean_sft_only=/tmp/sft_seed_{seed}/model",
        ],
        seed=23,
    )
    assert specs == [
        "defer_full=/tmp/defer_full_seed_23/model",
        "clean_sft_only=/tmp/sft_seed_23/model",
    ]


def test_format_policy_templates_rejects_static_paths_by_default() -> None:
    with pytest.raises(ValueError, match="must include '\\{seed\\}'"):
        _format_policy_templates(
            templates=["defer_full=/tmp/defer_full/model"],
            seed=23,
        )


def test_format_policy_templates_allows_static_paths_with_override() -> None:
    specs = _format_policy_templates(
        templates=["defer_full=/tmp/defer_full/model"],
        seed=23,
        allow_static_model_policy_templates=True,
    )
    assert specs == ["defer_full=/tmp/defer_full/model"]
