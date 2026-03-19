from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any


FAULT_TYPES = ["none", "timeout", "rate_limit", "partial_response", "schema_drift", "missing_field"]


@dataclass(frozen=True)
class FaultProfile:
    lambda_fault: float
    enable_schema_drift: bool = True
    enable_partial: bool = True
    enable_timeout: bool = True
    enable_rate_limit: bool = True
    enable_missing_field: bool = True


class FaultInjector:
    def __init__(self, profile: FaultProfile, seed: int = 0) -> None:
        self.profile = profile
        self._rng = random.Random(seed)

    def inject(
        self,
        tool_name: str,
        args: dict[str, Any],
        observation: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
        args = copy.deepcopy(args)
        obs = copy.deepcopy(observation) if observation is not None else None
        if self._rng.random() >= self.profile.lambda_fault:
            return args, obs, "none"

        candidates: list[str] = []
        if self.profile.enable_timeout:
            candidates.append("timeout")
        if self.profile.enable_rate_limit:
            candidates.append("rate_limit")
        if self.profile.enable_partial:
            candidates.append("partial_response")
        if self.profile.enable_schema_drift:
            candidates.append("schema_drift")
        if self.profile.enable_missing_field:
            candidates.append("missing_field")
        fault = self._rng.choice(candidates or ["none"])

        if fault == "schema_drift":
            if "start_time" in args:
                args["starts_at"] = args.pop("start_time")
            elif "payload" in args and isinstance(args["payload"], dict):
                args["data"] = args.pop("payload")
            if obs is not None and "status" in obs:
                obs["state"] = obs.pop("status")
        elif fault == "missing_field":
            for key in ["subject", "title", "table", "primary_key"]:
                if key in args:
                    del args[key]
                    break
        elif fault == "partial_response" and obs is not None:
            keys = list(obs.keys())
            if keys:
                keep = max(1, len(keys) // 2)
                obs = {k: obs[k] for k in sorted(keys)[:keep]}
        elif fault == "timeout":
            args["__fault_timeout__"] = True
        elif fault == "rate_limit":
            args["__fault_rate_limit__"] = True

        return args, obs, fault
