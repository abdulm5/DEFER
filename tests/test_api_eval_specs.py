from scripts.run_api_eval import _parse_api_specs


def test_parse_api_specs() -> None:
    specs = _parse_api_specs(
        [
            "frontier_gpt4o=gpt-4o",
            "frontier_sonnet=claude-3-7-sonnet-latest",
        ]
    )
    assert specs == [
        ("frontier_gpt4o", "gpt-4o"),
        ("frontier_sonnet", "claude-3-7-sonnet-latest"),
    ]
