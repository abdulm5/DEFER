from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolExecutionResult:
    ok: bool
    observation: dict[str, Any]
    state_diff: dict[str, Any]
    pending_fields: list[str]
    irreversible_commit: bool
