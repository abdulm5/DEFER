from __future__ import annotations

from defer.baselines.api_policy import OpenAIChatPolicy
from defer.baselines.prompted_deferral_policy import (
    PROMPTED_DEFERRAL_SYSTEM_PROMPT,
    PromptedDeferralPolicy,
)


def test_system_prompt_contains_key_deferral_keywords():
    keywords = ["defer", "confidence", "irreversible", "SAFE_COMMIT_REVERSIBLE", "cross-check", "fresh"]
    for keyword in keywords:
        assert keyword.lower() in PROMPTED_DEFERRAL_SYSTEM_PROMPT.lower(), (
            f"Missing keyword '{keyword}' in system prompt"
        )


def test_inherits_from_openai_chat_policy():
    assert issubclass(PromptedDeferralPolicy, OpenAIChatPolicy)


def test_fallback_uses_runtime_verified():
    import os
    os.environ.setdefault("OPENAI_API_KEY", "test-key-for-unit-test")
    try:
        policy = PromptedDeferralPolicy(model="gpt-4o")
        assert policy.fallback_policy.name == "runtime_verification_only"
        assert policy.name == "prompted_deferral"
        assert policy.system_prompt == PROMPTED_DEFERRAL_SYSTEM_PROMPT
    finally:
        if os.environ.get("OPENAI_API_KEY") == "test-key-for-unit-test":
            del os.environ["OPENAI_API_KEY"]
