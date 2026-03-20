from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from defer.core.contracts import check_postconditions, check_preconditions
from defer.core.interfaces import (
    ContradictionSource,
    ContractSpec,
    Freshness,
    PendingPostconditionReason,
    VerificationDecision,
    VerifierOutput,
)
from defer.core.verifier import UncertainVerifier, VerifierConfig


@dataclass(frozen=True)
class FailureProfile:
    stale_prob: float
    provisional_prob: float
    contradiction_prob: float


@dataclass(frozen=True)
class CorrelatedVerifierConfig:
    base_config: VerifierConfig
    failure_profiles: dict[tuple[str, str], FailureProfile] = field(default_factory=dict)
    consecutive_provisional_stale_boost: float = 0.12
    max_stale_boost: float = 0.40


class CorrelatedVerifier(UncertainVerifier):
    def __init__(
        self,
        config: CorrelatedVerifierConfig,
        seed: int = 0,
    ) -> None:
        super().__init__(config=config.base_config, seed=seed)
        self._correlated_config = config
        self._consecutive_provisionals: int = 0

    def verify(
        self,
        contract: ContractSpec,
        pre_state: dict[str, Any],
        post_state: dict[str, Any],
        pending_fields: list[str] | None = None,
        fault_mode: str | None = None,
        delayed_truth_category: str | None = None,
    ) -> VerifierOutput:
        pending_fields = pending_fields or []
        effective = self._effective_probabilities(fault_mode, delayed_truth_category)
        stale_prob = self._apply_consecutive_boost(effective.stale_prob)

        pre_failures = check_preconditions(contract, pre_state)
        post_failures = check_postconditions(contract, post_state)
        freshness = (
            Freshness.STALE if self._rng.random() < stale_prob else Freshness.FRESH
        )

        if pre_failures or post_failures:
            confidence = max(
                0.0, self.config.base_confidence - self.config.reject_penalty
            )
            if freshness == Freshness.STALE:
                confidence = max(0.0, confidence - self.config.stale_penalty)
            self._consecutive_provisionals = 0
            return VerifierOutput(
                decision=VerificationDecision.REJECT,
                confidence=round(confidence, 4),
                freshness=freshness,
                pending_postconditions=[],
                pending_postcondition_reason=None,
                evidence_ids=sorted(set(pre_failures + post_failures)),
            )

        if pending_fields or self._rng.random() < effective.provisional_prob:
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
            self._consecutive_provisionals += 1
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
        self._consecutive_provisionals = 0
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
        effective = self._effective_probabilities(fault_mode, delayed_truth_category)
        contradicted = self._rng.random() < effective.contradiction_prob
        if not contradicted:
            return False, None
        return True, self._sample_contradiction_source(
            fault_mode=fault_mode, delayed_truth_category=delayed_truth_category
        )

    def _effective_probabilities(
        self,
        fault_mode: str | None,
        delayed_truth_category: str | None,
    ) -> FailureProfile:
        key = (fault_mode or "none", delayed_truth_category or "A")
        profile = self._correlated_config.failure_profiles.get(key)
        if profile is not None:
            return profile
        return FailureProfile(
            stale_prob=self.config.stale_probability,
            provisional_prob=self.config.provisional_probability,
            contradiction_prob=self.config.contradiction_probability,
        )

    def _apply_consecutive_boost(self, base_stale_prob: float) -> float:
        boost = (
            self._correlated_config.consecutive_provisional_stale_boost
            * self._consecutive_provisionals
        )
        boost = min(boost, self._correlated_config.max_stale_boost)
        return min(0.95, base_stale_prob + boost)


def load_failure_profiles(path: Path) -> dict[tuple[str, str], FailureProfile]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    profiles: dict[tuple[str, str], FailureProfile] = {}
    for key_str, triple in raw.items():
        parts = key_str.split("|", 1)
        if len(parts) != 2:
            continue
        fault_mode, category = parts
        profiles[(fault_mode, category)] = FailureProfile(
            stale_prob=triple[0],
            provisional_prob=triple[1],
            contradiction_prob=triple[2],
        )
    return profiles
