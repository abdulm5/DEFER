from __future__ import annotations

import random


PARAPHRASE_MAP = {
    "schedule": ["arrange", "set up", "plan"],
    "email": ["message", "mail"],
    "urgent": ["high-priority", "time-sensitive"],
    "confirm": ["validate", "double-check"],
    "database": ["db", "data store"],
    "webhook": ["callback", "hook", "listener"],
    "upload": ["transfer", "push", "send"],
    "permission": ["access", "authorization", "privilege"],
    "notification": ["alert", "message", "ping"],
    "grant": ["allow", "authorize", "permit"],
    "file": ["document", "artifact", "attachment"],
    "deploy": ["release", "ship", "push"],
}


def perturb_prompt(prompt: str, epsilon: float, seed: int) -> str:
    """
    Deterministically apply semantic-preserving perturbations.
    """
    rng = random.Random(seed)
    words = prompt.split()
    out: list[str] = []
    for word in words:
        token = word.strip(".,!?").lower()
        if token in PARAPHRASE_MAP and rng.random() < epsilon:
            replacement = rng.choice(PARAPHRASE_MAP[token])
            out.append(replacement)
        else:
            out.append(word)
    if epsilon >= 0.2 and len(out) > 6:
        pivot = rng.randint(2, len(out) - 3)
        out = out[pivot:] + out[:pivot]
    if epsilon >= 0.3:
        out.append("Please keep constraints intact.")
    return " ".join(out)
