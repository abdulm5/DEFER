from __future__ import annotations

import json
import urllib.error
import urllib.request

from defer.baselines.api_policy import APIInferenceConfig, OpenAIChatPolicy
from defer.baselines.policies import RuntimeVerifiedPolicy


class _DummyResponse:
    def __init__(self, body: str, status: int = 200) -> None:
        self._body = body.encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _build_policy(*, auth_mode: str = "bearer", query_params: dict[str, str] | None = None) -> OpenAIChatPolicy:
    return OpenAIChatPolicy(
        name="api_test",
        model="fake-model",
        fallback_policy=RuntimeVerifiedPolicy(),
        api_key_env="OPENAI_API_KEY",
        base_url="https://example.test/v1/chat/completions?foo=1",
        inference=APIInferenceConfig(
            auth_mode=auth_mode,
            api_key_header="api-key",
            query_params=query_params,
            max_retries=2,
            retry_backoff_seconds=0.01,
            retry_max_backoff_seconds=0.01,
        ),
    )


def test_bearer_auth_and_usage_counters(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "token-123")
    captured: dict[str, object] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: int):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _DummyResponse(
            json.dumps(
                {
                    "choices": [{"message": {"content": "{\"ok\": true}"}}],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "total_tokens": 18,
                    },
                }
            ),
            status=200,
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    policy = _build_policy(auth_mode="bearer")
    text = policy._invoke("hello")
    stats = policy.stats()

    assert text == "{\"ok\": true}"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["authorization"] == "Bearer token-123"
    assert "api-key" not in headers
    assert stats["prompt_tokens_total"] == 11
    assert stats["completion_tokens_total"] == 7
    assert stats["total_tokens_total"] == 18


def test_api_key_mode_and_query_param_merge(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "api-key-token")
    captured: dict[str, object] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: int):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        return _DummyResponse(json.dumps({"choices": [{"message": {"content": "{\"x\":1}"}}]}))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    policy = _build_policy(auth_mode="api_key", query_params={"api-version": "2024-10-21", "foo": "2"})
    policy._invoke("hello")

    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["api-key"] == "api-key-token"
    assert "authorization" not in headers
    assert "api-version=2024-10-21" in str(captured["url"])
    assert "foo=2" in str(captured["url"])


def test_retries_on_429_and_parses_list_content(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "token")
    monkeypatch.setattr("defer.baselines.api_policy.time.sleep", lambda _: None)
    monkeypatch.setattr("defer.baselines.api_policy.random.uniform", lambda _a, _b: 0.0)
    attempts = {"count": 0}

    def fake_urlopen(req: urllib.request.Request, timeout: int):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "rate limited", {}, None)
        return _DummyResponse(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "text", "text": "line1"},
                                    {"type": "text", "text": "line2"},
                                ]
                            }
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    policy = _build_policy(auth_mode="bearer")
    text = policy._invoke("hello")
    stats = policy.stats()

    assert attempts["count"] == 2
    assert text == "line1\nline2"
    assert stats["http_retry_count"] == 1
    assert stats["http_429_count"] == 1


def test_retries_on_5xx(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "token")
    monkeypatch.setattr("defer.baselines.api_policy.time.sleep", lambda _: None)
    monkeypatch.setattr("defer.baselines.api_policy.random.uniform", lambda _a, _b: 0.0)
    attempts = {"count": 0}

    def fake_urlopen(req: urllib.request.Request, timeout: int):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "upstream error", {}, None)
        return _DummyResponse(json.dumps({"choices": [{"message": {"content": "ok"}}]}), status=200)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    policy = _build_policy(auth_mode="bearer")
    assert policy._invoke("hello") == "ok"
    assert policy.stats()["http_5xx_count"] == 1
