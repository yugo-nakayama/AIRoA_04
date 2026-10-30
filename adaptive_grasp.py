"""Experience-driven grasp candidate generation and selection.

The module deliberately learns *which grasp candidate* to use; it does not replace
Franka IK/PD motion control.  Each training row combines observable object features,
a proposed candidate, environment estimates and a binary physical success label.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURE_NAMES = (
    "class_circle", "class_square", "class_triangle",
    "confidence", "margin", "entropy", "novelty_score",
    "circularity", "solidity", "aspect", "vertices_norm",
    "height_m", "depth_mad_m", "edge_clearance_m",
    "friction_estimate", "position_sigma_m", "orientation_sigma_rad", "spacing_m",
    "candidate_dx_m", "candidate_dy_m", "candidate_yaw_offset_rad",
    "candidate_width_m", "candidate_force_n", "attempt_index",
)


@dataclass(frozen=True)
class GraspCandidate:
    candidate_id: str
    dx_m: float
    dy_m: float
    yaw_offset_rad: float
    width_m: float
    force_n: float

    def record(self) -> dict:
        return asdict(self)


def generate_candidates(observation: Mapping[str, float | str], attempt: int = 0) -> list[GraspCandidate]:
    """Generate diverse, bounded candidates in position, yaw, width and force."""
    label = str(observation.get("predicted_label", "circle"))
    base_width = {"circle": 0.048, "square": 0.050, "triangle": 0.044}.get(label, 0.047)
    base_force = {"circle": 1.6, "square": 1.8, "triangle": 1.45}.get(label, 1.7)
    shift = 0.0035 * min(attempt, 2)
    specs = [
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (+0.004 + shift, 0.0, 0.0, +0.003, +0.25),
        (-0.004 - shift, 0.0, 0.0, +0.003, +0.25),
        (0.0, +0.004 + shift, np.pi / 2, -0.002, +0.15),
        (0.0, -0.004 - shift, np.pi / 2, -0.002, +0.15),
        (+0.003, +0.003, np.pi / 4, +0.001, +0.45),
        (-0.003, -0.003, -np.pi / 4, +0.001, +0.45),
        (0.0, 0.0, np.pi / 2, +0.005, +0.65),
    ]
    candidates = []
    for i, (dx, dy, yaw, dw, df) in enumerate(specs):
        candidates.append(
            GraspCandidate(
                candidate_id=f"g{i}", dx_m=float(dx), dy_m=float(dy),
                yaw_offset_rad=float(yaw), width_m=float(np.clip(base_width + dw, 0.036, 0.070)),
                force_n=float(np.clip(base_force + df, 1.0, 3.0)),
            )
        )
    return candidates


def feature_vector(observation: Mapping[str, float | str], candidate: GraspCandidate, attempt: int) -> np.ndarray:
    label = str(observation.get("predicted_label", ""))
    vals = [
        label == "circle", label == "square", label == "triangle",
        observation.get("confidence", 0.0), observation.get("margin", 0.0),
        observation.get("entropy", 1.0), observation.get("novelty_score", 1.0),
        observation.get("circularity", 0.0), observation.get("solidity", 0.0),
        observation.get("aspect", 0.0), observation.get("vertices_norm", 0.0),
        observation.get("height_m", 0.0), observation.get("depth_mad_m", 0.0),
        observation.get("edge_clearance_m", 0.0), observation.get("friction_estimate", 0.55),
        observation.get("position_sigma_m", 0.006), observation.get("orientation_sigma_rad", 0.15),
        observation.get("spacing_m", 0.075), candidate.dx_m, candidate.dy_m,
        candidate.yaw_offset_rad, candidate.width_m, candidate.force_n, attempt,
    ]
    return np.asarray(vals, dtype=np.float32)


class SuccessPredictor:
    """Gradient-boosted binary grasp-success predictor with model metadata."""

    def __init__(self, random_state: int = 37):
        self.model = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.055, max_leaf_nodes=23,
            min_samples_leaf=25, l2_regularization=0.08, random_state=random_state,
        )
        self.metadata: dict = {"model_type": "HistGradientBoostingClassifier", "seed": random_state}
        self.fitted = False

    def fit(self, observations: Sequence[Mapping], candidates: Sequence[GraspCandidate], attempts: Sequence[int], labels: Sequence[int], **metadata) -> "SuccessPredictor":
        x = np.vstack([feature_vector(o, c, a) for o, c, a in zip(observations, candidates, attempts)])
        y = np.asarray(labels, np.int64)
        if len(np.unique(y)) < 2:
            raise ValueError("success predictor requires both success and failure examples")
        self.model.fit(x, y)
        self.fitted = True
        self.metadata.update({"feature_names": FEATURE_NAMES, "rows": int(len(y)), "positive_rate": float(y.mean()), **metadata})
        return self

    def predict_success(self, observation: Mapping, candidates: Sequence[GraspCandidate], attempt: int = 0) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("success predictor is not fitted")
        x = np.vstack([feature_vector(observation, c, attempt) for c in candidates])
        return self.model.predict_proba(x)[:, 1]

    def select(self, observation: Mapping, candidates: Sequence[GraspCandidate], attempt: int = 0, risk_penalty: float = 0.015) -> tuple[GraspCandidate, np.ndarray]:
        p = self.predict_success(observation, candidates, attempt)
        motion_cost = np.asarray([abs(c.dx_m) + abs(c.dy_m) + 0.003 * abs(c.yaw_offset_rad) + 0.003 * max(c.force_n - 1.5, 0) for c in candidates])
        utility = p - risk_penalty * motion_cost / 0.01
        index = int(np.argmax(utility))
        return candidates[index], p

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "metadata": self.metadata, "fitted": self.fitted}, path)

    @classmethod
    def load(cls, path: str | Path) -> "SuccessPredictor":
        bundle = joblib.load(path)
        obj = cls(int(bundle.get("metadata", {}).get("seed", 37)))
        obj.model, obj.metadata, obj.fitted = bundle["model"], bundle["metadata"], bool(bundle.get("fitted", True))
        return obj


def append_experience(path: str | Path, observation: Mapping, candidate: GraspCandidate, attempt: int, success: bool, failure: str, run_id: str) -> None:
    """Append a candidate/outcome row. JSONL keeps failed experiences auditable."""
    import json
    record = {"run_id": run_id, "observation": dict(observation), "candidate": candidate.record(), "attempt": int(attempt), "success": bool(success), "failure": str(failure)}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
