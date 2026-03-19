from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
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

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {api_key_env}")
        self.api_key = api_key

    def _invoke(self, user_prompt: str) -> str:
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
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.inference.timeout_seconds) as response:
            body = response.read().decode("utf-8")
        data = json.loads(body)
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        return content if isinstance(content, str) else ""

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
            "total_decisions": self.total_decisions,
            "parse_failures": self.parse_failures,
            "api_errors": self.api_errors,
            "fallback_calls": self.fallback_calls,
            "parse_failure_rate": self.parse_failures / max(1, self.total_decisions),
        }
