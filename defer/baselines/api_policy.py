from __future__ import annotations

import copy
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import Any

from defer.baselines.model_policy import parse_policy_decision_text
from defer.baselines.policies import Policy, PolicyDecision
from defer.core.interfaces import AgentAction


@dataclass(frozen=True)
class APIInferenceConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    timeout_seconds: int = 60
    auth_mode: str = "bearer"
    api_key_header: str = "api-key"
    extra_headers: dict[str, str] | None = None
    query_params: dict[str, str] | None = None
    extra_body: dict[str, Any] | None = None
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    retry_max_backoff_seconds: float = 8.0


def _build_prompt(context: dict[str, Any]) -> str:
    compact = {
        "user_prompt": context.get("prompt", ""),
        "required_tool": context.get("required_tool"),
        "tool_args": context.get("tool_args", {}),
        "unresolved_truth": bool(context.get("unresolved_truth", False)),
        "last_verifier_decision": str(context.get("last_verifier_decision")),
        "last_freshness": context.get("last_freshness"),
        "last_confidence": context.get("last_confidence"),
        "last_pending_reason": context.get("last_pending_reason"),
        "fault_seen": bool(context.get("fault_seen", False)),
        "has_delayed_truth": bool(context.get("has_delayed_truth", False)),
        "scenario_category": context.get("scenario_category"),
        "delay_mechanism": context.get("delay_mechanism"),
        "expects_irreversible": bool(context.get("expects_irreversible", False)),
        "requires_refresh": bool(context.get("requires_refresh", False)),
        "turn_idx": int(context.get("turn_idx", 0)),
        "remaining_turns": int(context.get("remaining_turns", 1)),
        "domain": context.get("domain"),
    }
    return (
        "You are a tool-using DEFER agent.\n"
        "Return exactly one JSON object with keys: action, tool_name, tool_args.\n"
        f"Valid action values: {', '.join([action.value for action in AgentAction])}.\n"
        "If action is DEFER_WAIT or DEFER_ASK_USER, set tool_name null and tool_args {}.\n"
        "If action is DEFER_REFRESH or CROSS_CHECK_SECOND_TOOL, use tool_name refresh_state.\n"
        "Do not include explanations.\n"
        f"Context: {json.dumps(compact, sort_keys=True)}\n"
        "JSON:"
    )


class OpenAIChatPolicy:
    def __init__(
        self,
        name: str,
        model: str,
        fallback_policy: Policy,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1/chat/completions",
        system_prompt: str = "You are a reliable tool-using planning policy.",
        inference: APIInferenceConfig | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.fallback_policy = fallback_policy
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.inference = inference or APIInferenceConfig()
        self.total_decisions = 0
        self.parse_failures = 0
        self.api_errors = 0
        self.fallback_calls = 0
        self.http_requests_total = 0
        self.http_retry_count = 0
        self.http_429_count = 0
        self.http_5xx_count = 0
        self.last_http_status: int | None = None
        self.prompt_tokens_total = 0
        self.completion_tokens_total = 0
        self.total_tokens_total = 0

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {api_key_env}")
        self.api_key = api_key
        if self.inference.auth_mode not in {"bearer", "api_key"}:
            raise ValueError(
                f"Unsupported auth_mode '{self.inference.auth_mode}'. "
                "Expected one of: bearer, api_key."
            )
        if self.inference.auth_mode == "api_key" and not self.inference.api_key_header:
            raise ValueError("api_key auth_mode requires a non-empty api_key_header.")

    def _compose_url(self) -> str:
        parts = urlsplit(self.base_url)
        existing = dict(parse_qsl(parts.query, keep_blank_values=True))
        for key, value in (self.inference.query_params or {}).items():
            existing[str(key)] = str(value)
        new_query = urlencode(existing)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))

    @staticmethod
    def _extract_content(message_content: Any) -> str:
        if isinstance(message_content, str):
            return message_content
        if isinstance(message_content, list):
            chunks: list[str] = []
            for item in message_content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "\n".join(chunks).strip()
        return ""

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.inference.auth_mode == "api_key":
            headers[self.inference.api_key_header] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for key, value in (self.inference.extra_headers or {}).items():
            headers[str(key)] = str(value)
        return headers

    def _register_usage(self, data: dict[str, Any]) -> None:
        usage = data.get("usage", {})
        if not isinstance(usage, dict):
            return
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if prompt_tokens is None:
            prompt_tokens = usage.get("input_tokens")
        if completion_tokens is None:
            completion_tokens = usage.get("output_tokens")
        if total_tokens is None and isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = prompt_tokens + completion_tokens
        if isinstance(prompt_tokens, int):
            self.prompt_tokens_total += prompt_tokens
        if isinstance(completion_tokens, int):
            self.completion_tokens_total += completion_tokens
        if isinstance(total_tokens, int):
            self.total_tokens_total += total_tokens

    def _should_retry(self, status_code: int) -> bool:
        return status_code == 429 or 500 <= status_code < 600

    def _invoke(self, user_prompt: str) -> str:
        path = urlsplit(self.base_url).path
        uses_anthropic_messages = path.endswith("/anthropic/v1/messages")
        if uses_anthropic_messages:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": self.inference.temperature,
                "max_tokens": self.inference.max_new_tokens,
                "stream": False,
            }
            if self.system_prompt:
                payload["system"] = self.system_prompt
        else:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.inference.temperature,
                "top_p": self.inference.top_p,
                "max_tokens": self.inference.max_new_tokens,
            }
        if self.inference.extra_body:
            for key, value in self.inference.extra_body.items():
                payload[key] = copy.deepcopy(value)
        url = self._compose_url()
        max_attempts = max(1, int(self.inference.max_retries) + 1)
        request_data = json.dumps(payload).encode("utf-8")

        for attempt in range(max_attempts):
            req = urllib.request.Request(
                url,
                data=request_data,
                headers=self._headers(),
                method="POST",
            )
            self.http_requests_total += 1
            try:
                with urllib.request.urlopen(req, timeout=self.inference.timeout_seconds) as response:
                    self.last_http_status = int(response.status)
                    body = response.read().decode("utf-8")
                data = json.loads(body)
                self._register_usage(data)
                choices = data.get("choices", [])
                if not choices:
                    content = data.get("content")
                    return self._extract_content(content)
                message = choices[0].get("message", {})
                return self._extract_content(message.get("content", ""))
            except urllib.error.HTTPError as exc:
                self.last_http_status = int(exc.code)
                if exc.code == 429:
                    self.http_429_count += 1
                if 500 <= exc.code < 600:
                    self.http_5xx_count += 1
                if attempt >= max_attempts - 1 or not self._should_retry(exc.code):
                    raise
                self.http_retry_count += 1
                backoff = min(
                    self.inference.retry_max_backoff_seconds,
                    self.inference.retry_backoff_seconds * (2**attempt),
                )
                time.sleep(backoff + random.uniform(0.0, min(0.25, backoff * 0.1)))
            except urllib.error.URLError:
                self.last_http_status = None
                if attempt >= max_attempts - 1:
                    raise
                self.http_retry_count += 1
                backoff = min(
                    self.inference.retry_max_backoff_seconds,
                    self.inference.retry_backoff_seconds * (2**attempt),
                )
                time.sleep(backoff + random.uniform(0.0, min(0.25, backoff * 0.1)))
        return ""

    def decide(self, context: dict[str, Any]) -> PolicyDecision:
        self.total_decisions += 1
        prompt = _build_prompt(context)
        completion = ""
        try:
            completion = self._invoke(prompt)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            self.api_errors += 1
        decision = parse_policy_decision_text(completion, context=context)
        if decision is not None:
            return decision
        self.parse_failures += 1
        self.fallback_calls += 1
        fallback = self.fallback_policy.decide(context)
        return PolicyDecision(
            action=fallback.action,
            tool_name=fallback.tool_name,
            tool_args=fallback.tool_args,
            reason=f"api_fallback::{fallback.reason}",
        )

    def stats(self) -> dict[str, Any]:
        return {
            "policy_name": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "auth_mode": self.inference.auth_mode,
            "api_key_header": self.inference.api_key_header,
            "extra_headers": self.inference.extra_headers or {},
            "query_params": self.inference.query_params or {},
            "extra_body": self.inference.extra_body or {},
            "total_decisions": self.total_decisions,
            "parse_failures": self.parse_failures,
            "api_errors": self.api_errors,
            "fallback_calls": self.fallback_calls,
            "parse_failure_rate": self.parse_failures / max(1, self.total_decisions),
            "http_requests_total": self.http_requests_total,
            "http_retry_count": self.http_retry_count,
            "http_429_count": self.http_429_count,
            "http_5xx_count": self.http_5xx_count,
            "last_http_status": self.last_http_status,
            "prompt_tokens_total": self.prompt_tokens_total,
            "completion_tokens_total": self.completion_tokens_total,
            "total_tokens_total": self.total_tokens_total,
        }
