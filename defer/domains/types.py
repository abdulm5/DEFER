from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass
from typing import Any


_tool_call_counter = threading.local()


def deterministic_id(prefix: str) -> str:
    """Generate a reproducible ID when called in a deterministic order.

    Falls back to uuid4 if the counter has not been initialised via
    ``reset_tool_call_counter``.
    """
    if not hasattr(_tool_call_counter, "value"):
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    _tool_call_counter.value += 1
    digest = hashlib.sha1(
        f"{prefix}:{_tool_call_counter.value}:{_tool_call_counter.seed}".encode()
    ).hexdigest()
    return f"{prefix}_{digest[:8]}"


def reset_tool_call_counter(seed: int) -> None:
    """Reset the per-thread deterministic counter.  Call at episode start."""
    _tool_call_counter.value = 0
    _tool_call_counter.seed = seed


@dataclass
class ToolExecutionResult:
    ok: bool
    observation: dict[str, Any]
    state_diff: dict[str, Any]
    pending_fields: list[str]
    irreversible_commit: bool
