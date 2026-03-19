from __future__ import annotations

from pydantic import BaseModel, Field


class Scenario(BaseModel):
    scenario_id: str
    domain: str
    prompt: str
    required_tool: str
    tool_args: dict
    expects_irreversible: bool = False
    requires_refresh: bool = False
    has_delayed_truth: bool = True
    delayed_truth_category: str = "A"
    delay_mechanism: str = "none"
    contradiction_probability_override: float | None = None
    metadata: dict = Field(default_factory=dict)
