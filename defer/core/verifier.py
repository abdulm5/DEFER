from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from defer.core.contracts import check_postconditions, check_preconditions
from defer.core.interfaces import (
    ContradictionSource,
    ContractSpec,
    Freshness,
    PendingPostconditionReason,
    VerifierOutput,
    VerificationDecision,
)


@dataclass(frozen=True)
class VerifierConfig:
    stale_probability: float = 0.2
    provisional_probability: float = 0.25
    contradiction_probability: float = 0.15
    base_confidence: float = 0.9
    stale_penalty: float = 0.25
    provisional_penalty: float = 0.2
    reject_penalty: float = 0.5


class UncertainVerifier:
    def __init__(self, config: VerifierConfig | None = None, seed: int = 0) -> None:
        self.config = config or VerifierConfig()
        self._rng = random.Random(seed)

    def verify(
        self,
        contract: ContractSpec,
        pre_state: dict[str, Any],
        post_state: dict[str, Any],
        pending_fields: list[str] | None = None,
        fault_mode: str | None = None,
    ) -> VerifierOutput:
        pending_fields = pending_fields or []

        pre_failures = check_preconditions(contract, pre_state)
        post_failures = check_postconditions(contract, post_state)
        freshness = self._sample_freshness()

        if pre_failures or post_failures:
            confidence = max(0.0, self.config.base_confidence - self.config.reject_penalty)
            if freshness == Freshness.STALE:
                confidence = max(0.0, confidence - self.config.stale_penalty)
            return VerifierOutput(
                decision=VerificationDecision.REJECT,
                confidence=round(confidence, 4),
                freshness=freshness,
                pending_postconditions=[],
                pending_postcondition_reason=None,
                evidence_ids=sorted(set(pre_failures + post_failures)),
            )

        if pending_fields or self._rng.random() < self.config.provisional_probability:
            reason = self._provisional_reason(
                pending_fields=pending_fields,
                freshness=freshness,
                fault_mode=fault_mode,
            )
            confidence = max(
                0.0,
                self.config.base_confidence
                - self.config.provisional_penalty
                - (self.config.stale_penalty if freshness == Freshness.STALE else 0.0),
            )
            return VerifierOutput(
                decision=VerificationDecision.PROVISIONAL,
                confidence=round(confidence, 4),
                freshness=freshness,
                pending_postconditions=pending_fields,
                pending_postcondition_reason=reason,
                evidence_ids=[f"pending:{field}" for field in pending_fields],
            )

        confidence = self.config.base_confidence
        if freshness == Freshness.STALE:
            confidence = max(0.0, confidence - self.config.stale_penalty)
        return VerifierOutput(
            decision=VerificationDecision.ACCEPT,
            confidence=round(confidence, 4),
            freshness=freshness,
            pending_postconditions=[],
            pending_postcondition_reason=None,
            evidence_ids=["verified:postconditions"],
        )

    def maybe_contradict(
        self,
        fault_mode: str | None = None,
        delayed_truth_category: str | None = None,
    ) -> tuple[bool, ContradictionSource | None]:
        contradicted = self._rng.random() < self.config.contradiction_probability
        if not contradicted:
            return False, None
        return True, self._sample_contradiction_source(
            fault_mode=fault_mode, delayed_truth_category=delayed_truth_category
        )

    def _sample_freshness(self) -> Freshness:
        return (
            Freshness.STALE
            if self._rng.random() < self.config.stale_probability
            else Freshness.FRESH
        )

    def _provisional_reason(
        self,
        pending_fields: list[str],
        freshness: Freshness,
        fault_mode: str | None,
    ) -> PendingPostconditionReason:
        if pending_fields:
            return PendingPostconditionReason.DELAYED_SIDE_EFFECT_NOT_OBSERVED
        if fault_mode == "schema_drift":
            return PendingPostconditionReason.STALE_SCHEMA
        if fault_mode == "partial_response":
            return PendingPostconditionReason.PARTIAL_OUTPUT
        if freshness == Freshness.STALE:
            return PendingPostconditionReason.STALE_SCHEMA
        if fault_mode in {"rate_limit", "timeout"}:
            return PendingPostconditionReason.EXTERNAL_STATE_MAY_CHANGE
        return PendingPostconditionReason.UNKNOWN

    def _sample_contradiction_source(
        self,
        fault_mode: str | None,
        delayed_truth_category: str | None,
    ) -> ContradictionSource:
        if fault_mode == "schema_drift":
            return ContradictionSource.SCHEMA_CONFLICT
        if fault_mode == "partial_response":
            return ContradictionSource.CROSS_TOOL_MISMATCH
        if fault_mode in {"timeout", "rate_limit"}:
            return ContradictionSource.DELAYED_JOB_RESOLUTION
        if delayed_truth_category == "C":
            return self._rng.choice(
                [
                    ContradictionSource.CONCURRENT_EDIT,
                    ContradictionSource.CROSS_TOOL_MISMATCH,
                    ContradictionSource.STALE_CACHE_REVEAL,
                ]
            )
        if delayed_truth_category == "B":
            return self._rng.choice(
                [
                    ContradictionSource.DELAYED_JOB_RESOLUTION,
                    ContradictionSource.STALE_CACHE_REVEAL,
                ]
            )
        return ContradictionSource.UNKNOWN
