"""Risk-aware decision among grasp, arm-retreat re-observation, and hold."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class Action(str, Enum):
    GRASP = "grasp"
    REOBSERVE = "reobserve"
    HOLD = "hold"


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    expected_success: float


def decide(observation: Mapping, predicted_success: float, reobservations: int, enable_reobserve: bool, enable_unknown_hold: bool, max_reobservations: int = 1) -> Decision:
    if enable_unknown_hold and bool(observation.get("unknown", False)):
        return Decision(Action.HOLD, "geometry outside known-shape support", predicted_success)
    confidence = float(observation.get("confidence", 0.0))
    entropy = float(observation.get("entropy", 1.0))
    depth_mad = float(observation.get("depth_mad_m", 1.0))
    poor_view = confidence < 0.76 or entropy > 0.60 or depth_mad > 0.0045 or predicted_success < 0.52
    if enable_reobserve and poor_view and reobservations < max_reobservations:
        return Decision(Action.REOBSERVE, "retreat arm and reacquire RGB-D", predicted_success)
    if predicted_success < 0.34:
        return Decision(Action.HOLD, "all grasp candidates below safety utility", predicted_success)
    return Decision(Action.GRASP, "best candidate exceeds grasp threshold", predicted_success)
