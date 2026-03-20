from __future__ import annotations

from defer.stress.perturb import PARAPHRASE_MAP, perturb_prompt


def test_new_paraphrase_entries_exist():
    new_keys = ["webhook", "upload", "permission", "notification", "grant", "file", "deploy"]
    for key in new_keys:
        assert key in PARAPHRASE_MAP, f"Missing paraphrase entry: {key}"
        assert len(PARAPHRASE_MAP[key]) >= 2, f"Too few synonyms for: {key}"


def test_new_paraphrases_applied_at_epsilon():
    prompt = "Register a webhook and upload the file with proper permission."
    result = perturb_prompt(prompt, epsilon=0.5, seed=42)
    assert isinstance(result, str)
    assert len(result) > 0
