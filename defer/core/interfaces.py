from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SideEffectType(str, Enum):
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


class Freshness(str, Enum):
    FRESH = "fresh"
    STALE = "stale"


class VerificationDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    PROVISIONAL = "provisional"


class AgentAction(str, Enum):
    DEFER_WAIT = "DEFER_WAIT"
    DEFER_REFRESH = "DEFER_REFRESH"
    DEFER_ASK_USER = "DEFER_ASK_USER"
    CROSS_CHECK_SECOND_TOOL = "CROSS_CHECK_SECOND_TOOL"
    SAFE_COMMIT_REVERSIBLE = "SAFE_COMMIT_REVERSIBLE"
    FULL_COMMIT_IRREVERSIBLE = "FULL_COMMIT_IRREVERSIBLE"


class PendingPostconditionReason(str, Enum):
    DELAYED_SIDE_EFFECT_NOT_OBSERVED = "delayed_side_effect_not_observed"
    STALE_SCHEMA = "stale_schema"
    PARTIAL_OUTPUT = "partial_output"
    EXTERNAL_STATE_MAY_CHANGE = "external_state_may_change"
    UNKNOWN = "unknown"


class ContradictionSource(str, Enum):
    DELAYED_JOB_RESOLUTION = "delayed_job_resolution"
    SCHEMA_CONFLICT = "schema_conflict"
    CONCURRENT_EDIT = "concurrent_edit"
    CROSS_TOOL_MISMATCH = "cross_tool_mismatch"
    STALE_CACHE_REVEAL = "stale_cache_reveal"
    UNKNOWN = "unknown"


class ConditionSpec(BaseModel):
    field: str
    op: str = Field(description="eq|ne|gt|gte|lt|lte|in|exists")
    value: Any | None = None


class ContractSpec(BaseModel):
    tool_name: str
    preconditions: list[ConditionSpec]
    postconditions: list[ConditionSpec]
    side_effect_type: SideEffectType
    refresh_tools: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)


class VerifierOutput(BaseModel):
    decision: VerificationDecision
    confidence: float = Field(ge=0.0, le=1.0)
    freshness: Freshness
    pending_postconditions: list[str] = Field(default_factory=list)
    pending_postcondition_reason: PendingPostconditionReason | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    tool_name: str
    args: dict[str, Any]


class DelayedRevealEvent(BaseModel):
    event_id: str
    time_step: int
    description: str
    revealed_truth: dict[str, Any]
    contradiction: bool = False
    contradiction_source: ContradictionSource | None = None


class ProcedureGates(BaseModel):
    policy_compliance: int = 1
    policy_faithfulness: int = 1
    execution_consistency: int = 1
    data_faithfulness: int = 1
    intent_adherence: int = 1
    question_fulfillment: int = 1

    def all_pass(self) -> bool:
        return (
            self.policy_compliance
            * self.policy_faithfulness
            * self.execution_consistency
            * self.data_faithfulness
            * self.intent_adherence
            * self.question_fulfillment
            == 1
        )


class EpisodeTurn(BaseModel):
    turn_id: int
    prompt: str
    selected_action: AgentAction
    tool_call: ToolCall | None = None
    verifier_output: VerifierOutput | None = None
    observation: dict[str, Any] = Field(default_factory=dict)
    state_diff: dict[str, Any] = Field(default_factory=dict)
    retries: int = 0
    refusal: bool = False
    used_stale_evidence: bool = False
    unresolved_truth: bool = False
    irreversible_commit: bool = False
    contradiction_observed: bool = False
    contradiction_source: ContradictionSource | None = None


class EpisodeResult(BaseModel):
    success: bool
    invalid_commit: bool
    corrupt_success: bool
    unsafe_retry_count: int = 0
    turn_budget_exhausted: bool = False
    explanation: str = ""
    procedure_gates: ProcedureGates = Field(default_factory=ProcedureGates)


class EpisodeTrace(BaseModel):
    episode_id: str
    scenario_id: str
    domain: str
    delay_mechanism: str = "none"
    policy_name: str
    seed: int
    epsilon: float
    lambda_fault: float
    repeat_index: int
    scenario_category: str = "unknown"
    turns: list[EpisodeTurn]
    delayed_events: list[DelayedRevealEvent] = Field(default_factory=list)
    final_state: dict[str, Any] = Field(default_factory=dict)
    result: EpisodeResult


class ReliabilityRecord(BaseModel):
    episode_id: str
    scenario_id: str
    domain: str = "unknown"
    delay_mechanism: str = "none"
    policy_name: str
    seed: int
    k: int
    epsilon: float
    lambda_fault: float
    success: int
    gated_success: int
    corrupt_success: int
    invalid_commit: int
    deferred_when_unresolved: int
    deferred_when_resolved: int = 0
    committed_when_resolved: int
    committed_when_unresolved: int = 0
    unresolved_events: int
    resolved_events: int
    total_deferral_actions: int = 0
    total_commit_actions: int = 0
    over_deferrals: int = 0
    irreversible_errors: int
    evidence_freshness_violations: int
    delayed_contradictions: int
    turn_budget_exhausted: int = 0
    scenario_category: str = "unknown"
