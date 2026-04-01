from scripts.run_api_eval import _parse_api_specs, _parse_query_params


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


def test_parse_query_params() -> None:
    params = _parse_query_params(["api-version=2024-10-21", "foo=bar"])
    assert params == {"api-version": "2024-10-21", "foo": "bar"}
