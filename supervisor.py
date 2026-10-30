"""Decision, verification, and recovery logic for closed-loop sorting."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np

from perception import Detection


class Phase(str, Enum):
    OBSERVE = "observe"
    SELECT = "select"
    APPROACH = "approach"
    REALIGN = "realign"
    DESCEND = "descend"
    GRASP = "grasp"
    VERIFY = "verify"
    LIFT = "lift"
    TRANSPORT = "transport"
    PLACE = "place"
    RETREAT = "retreat"
    RECOVER = "recover"
    DONE = "done"
    ABORT = "abort"


class Failure(str, Enum):
    NONE = "none"
    NO_TARGET = "no_target"
    UNCERTAIN_CLASS = "uncertain_class"
    TARGET_LOST = "target_lost"
    IK_FAILED = "ik_failed"
    EMPTY_GRASP = "empty_grasp"
    SLIP = "slip"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class SelectionPolicy:
    min_confidence: float = 0.72
    min_margin: float = 0.24
    max_entropy: float = 0.68
    max_depth_mad_m: float = 0.006
    workspace_x: tuple[float, float] = (0.36, 0.70)
    workspace_y: tuple[float, float] = (-0.27, 0.29)

    def eligible(self, detection: Detection) -> bool:
        x, y = detection.position_world[:2]
        return (
            detection.usable
            and self.workspace_x[0] <= x <= self.workspace_x[1]
            and self.workspace_y[0] <= y <= self.workspace_y[1]
            and detection.confidence >= self.min_confidence
            and detection.margin >= self.min_margin
            and detection.entropy <= self.max_entropy
            and detection.depth_mad_m <= self.max_depth_mad_m
        )

    def score(self, detection: Detection, home_xy: Iterable[float] = (0.48, 0.0)) -> float:
        distance = np.linalg.norm(detection.position_world[:2] - np.asarray(home_xy, float))
        return float(
            1.20 * detection.confidence
            + 0.45 * detection.margin
            - 0.35 * detection.entropy
            - 0.8 * detection.depth_mad_m
            - 0.10 * distance
        )

    def select(self, detections: list[Detection]) -> Detection | None:
        eligible = [d for d in detections if self.eligible(d)]
        return max(eligible, key=self.score) if eligible else None


@dataclass(frozen=True)
class VerificationPolicy:
    min_finger_width_m: float = 0.0055
    min_lift_height_delta_m: float = 0.055
    max_width_drop_m: float = 0.004

    def finger_has_object(self, finger_width_m: float) -> bool:
        return float(finger_width_m) >= self.min_finger_width_m

    def lifted(self, before_height_m: float, after_height_m: float | None) -> bool:
        return after_height_m is not None and (after_height_m - before_height_m) >= self.min_lift_height_delta_m

    def slipped(self, width_at_grasp_m: float, width_after_motion_m: float) -> bool:
        return (width_at_grasp_m - width_after_motion_m) > self.max_width_drop_m

    def success(
        self,
        finger_width_m: float,
        before_height_m: float,
        after_height_m: float | None,
    ) -> bool:
        # Two independent signals are required: proprioception and post-lift RGB-D.
        return self.finger_has_object(finger_width_m) and self.lifted(before_height_m, after_height_m)


@dataclass
class RecoveryPlan:
    retry: bool
    xy_offset_m: tuple[float, float] = (0.0, 0.0)
    yaw_offset_rad: float = 0.0
    force_scale: float = 1.0
    retreat_height_m: float = 0.30
    reason: str = ""


@dataclass
class RecoveryPolicy:
    max_attempts_per_object: int = 3
    _offsets: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.006, 0.0), (-0.004, 0.006))

    def plan(self, failure: Failure, attempt: int) -> RecoveryPlan:
        if attempt >= self.max_attempts_per_object:
            return RecoveryPlan(False, reason="retry budget exhausted")
        offset = self._offsets[min(attempt, len(self._offsets) - 1)]
        if failure == Failure.UNCERTAIN_CLASS:
            return RecoveryPlan(False, reason="leave ambiguous object for later observation")
        if failure == Failure.TARGET_LOST:
            return RecoveryPlan(True, xy_offset_m=offset, reason="return home and reacquire")
        if failure == Failure.IK_FAILED:
            return RecoveryPlan(True, xy_offset_m=offset, retreat_height_m=0.36, reason="higher retreat, alternate approach")
        if failure == Failure.EMPTY_GRASP:
            return RecoveryPlan(True, xy_offset_m=offset, yaw_offset_rad=np.pi / 2, force_scale=1.12, reason="offset grasp and rotate jaws")
        if failure == Failure.SLIP:
            return RecoveryPlan(True, xy_offset_m=offset, force_scale=1.20, reason="increase force within safety limit")
        if failure == Failure.TIMEOUT:
            return RecoveryPlan(True, retreat_height_m=0.36, reason="reset from known safe pose")
        return RecoveryPlan(False, reason="failure is not recoverable")


@dataclass
class EpisodeState:
    phase: Phase = Phase.OBSERVE
    object_attempts: dict[str, int] = field(default_factory=dict)
    transitions: list[dict[str, str]] = field(default_factory=list)

    def transition(self, next_phase: Phase, reason: str = "") -> None:
        self.transitions.append({"from": self.phase.value, "to": next_phase.value, "reason": reason})
        self.phase = next_phase

    def attempt(self, key: str) -> int:
        current = self.object_attempts.get(key, 0)
        self.object_attempts[key] = current + 1
        return current
