from __future__ import annotations

import json
from typing import Any

from defer.core.interfaces import EpisodeTrace, EpisodeTurn


def render_turn(turn: EpisodeTurn | dict[str, Any]) -> str:
    row = turn if isinstance(turn, dict) else turn.model_dump(mode="json")
    compact = {
        "turn_id": row.get("turn_id"),
        "action": row.get("selected_action"),
        "tool_call": row.get("tool_call"),
        "verifier": row.get("verifier_output"),
        "observation": row.get("observation"),
        "state_diff": row.get("state_diff"),
        "unresolved_truth": row.get("unresolved_truth"),
        "irreversible_commit": row.get("irreversible_commit"),
        "used_stale_evidence": row.get("used_stale_evidence"),
        "contradiction_observed": row.get("contradiction_observed"),
        "contradiction_source": row.get("contradiction_source"),
    }
    return json.dumps(compact, sort_keys=True)


def render_trace_response(trace: EpisodeTrace) -> str:
    lines = [render_turn(turn) for turn in trace.turns]
    return "\n".join(lines)


def render_pair_response(turns: list[dict[str, Any]]) -> str:
    lines = [render_turn(turn) for turn in turns]
    return "\n".join(lines)
