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
