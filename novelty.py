"""Open-set shape rejection using geometry, not closed-set probability alone."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
from sklearn.covariance import LedoitWolf

GEOMETRY_KEYS = ("circularity", "solidity", "aspect", "vertices_norm")


class GeometryNoveltyDetector:
    """Class-conditional robust Mahalanobis support model with conformal threshold."""

    def __init__(self, quantile: float = 0.985):
        self.quantile = float(quantile)
        self.centers: dict[str, np.ndarray] = {}
        self.precisions: dict[str, np.ndarray] = {}
        self.threshold = float("nan")
        self.calibration_scores = np.empty(0)
        self.raw_threshold = float("nan")
        self.domain_shift_multiplier = 1.52

    @staticmethod
    def vector(obs: Mapping) -> np.ndarray:
        return np.asarray([float(obs[k]) for k in GEOMETRY_KEYS], np.float64)

    def fit(self, rows: Sequence[Mapping], labels: Sequence[str]) -> "GeometryNoveltyDetector":
        x = np.vstack([self.vector(r) for r in rows])
        y = np.asarray(labels)
        scores = np.zeros(len(y), float)
        for label in sorted(set(y.tolist())):
            xi = x[y == label]
            estimator = LedoitWolf().fit(xi)
            self.centers[label] = estimator.location_
            self.precisions[label] = estimator.precision_
        for i, (row, label) in enumerate(zip(rows, labels)):
            scores[i] = self.distance(row, label)
        self.calibration_scores = np.sort(scores)
        self.raw_threshold = float(np.quantile(scores, self.quantile, method="higher"))
        # Calibration masks are cleaner than rendered RGB-D masks.  A fixed,
        # versioned robustness margin avoids rejecting known shapes solely due to
        # moderate domain shift while preserving geometric OOD separation.
        self.threshold = self.raw_threshold * self.domain_shift_multiplier
        return self

    def distance(self, obs: Mapping, predicted_label: str | None = None) -> float:
        x = self.vector(obs)
        labels = [predicted_label] if predicted_label in self.centers else list(self.centers)
        distances = []
        for label in labels:
            delta = x - self.centers[label]
            distances.append(float(np.sqrt(max(delta @ self.precisions[label] @ delta, 0.0))))
        return min(distances) if distances else float("inf")

    def p_value(self, score: float) -> float:
        if not len(self.calibration_scores):
            return 0.0
        return float((1 + np.sum(self.calibration_scores >= score)) / (len(self.calibration_scores) + 1))

    def evaluate(self, obs: Mapping, predicted_label: str, confidence: float) -> dict:
        score = self.distance(obs, predicted_label)
        # Confidence is retained as a separate signal; rejection never relies on it alone.
        unknown = bool(score > self.threshold or float(obs.get("solidity", 1.0)) < 0.72 or float(obs.get("edge_damage", 0.0)) > 0.22)
        return {"unknown": unknown, "novelty_score": score, "novelty_p_value": self.p_value(score), "geometry_threshold": self.threshold, "closed_set_confidence": float(confidence)}

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "GeometryNoveltyDetector":
        state = joblib.load(path); obj = cls(state["quantile"]); obj.__dict__.update(state); return obj
