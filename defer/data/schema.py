from __future__ import annotations

from pydantic import BaseModel, Field


class SeedTask(BaseModel):
    task_id: str
    domain: str
    template_id: str
    split: str
    prompt: str
    required_tool: str
    tool_args: dict
    expects_irreversible: bool = False
    requires_refresh: bool = False


class VariantTask(BaseModel):
    variant_id: str
    task_id: str
    domain: str
    split: str
    prompt: str
    required_tool: str
    tool_args: dict
    epsilon: float
    lambda_fault: float
    delay_setting: str
    delayed_truth_category: str = "A"
    delay_mechanism: str = "none"
    fault_profile_id: str
    expects_irreversible: bool = False
    requires_refresh: bool = False
    metadata: dict = Field(default_factory=dict)
