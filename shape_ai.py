"""Uncertainty-aware silhouette classifier for the advanced RGB-D sorter.

The model is deliberately independent from Genesis so it can be trained and tested
on a CPU.  It combines a normalized 32x32 silhouette with geometric descriptors,
and averages predictions over small morphology perturbations (test-time
augmentation).  The returned entropy and probability margin let the robot reject
ambiguous observations instead of confidently sorting them into the wrong bin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

CLASSES = ("circle", "square", "triangle")
PATCH = 32
MODEL_VERSION = 2


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("mask does not contain a foreground object")
    return max(contours, key=cv2.contourArea)


def normalized_patch(mask: np.ndarray, patch: int = PATCH) -> np.ndarray:
    """Return a centered, scale-normalized binary patch in [0, 1]."""
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    ys, xs = np.nonzero(m)
    if len(xs) < 8:
        raise ValueError("foreground is too small")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = max(int(np.ceil(max(x1 - x0 + 1, y1 - y0 + 1) * 1.25)), 8)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    x0c, y0c = int(round(cx - side / 2)), int(round(cy - side / 2))
    canvas = np.zeros((side, side), np.uint8)
    xa0, ya0 = max(0, x0c), max(0, y0c)
    xa1, ya1 = min(m.shape[1], x0c + side), min(m.shape[0], y0c + side)
    if xa1 > xa0 and ya1 > ya0:
        canvas[ya0 - y0c : ya1 - y0c, xa0 - x0c : xa1 - x0c] = m[ya0:ya1, xa0:xa1]
    return cv2.resize(canvas, (patch, patch), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def geometric_features(mask: np.ndarray) -> np.ndarray:
    """Compact, interpretable descriptors appended to the pixel feature."""
    contour = _largest_contour(mask)
    area = max(float(cv2.contourArea(contour)), 1.0)
    perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
    x, y, w, h = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    min_side, max_side = max(min(rw, rh), 1e-6), max(rw, rh, 1e-6)
    eps = 0.035 * perimeter
    vertices = len(cv2.approxPolyDP(contour, eps, True))
    hu = cv2.HuMoments(cv2.moments(contour)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    values = np.array(
        [
            4.0 * np.pi * area / (perimeter * perimeter),
            area / hull_area,
            area / max(float(w * h), 1.0),
            min_side / max_side,
            np.clip(vertices / 8.0, 0.0, 1.5),
            *np.clip(hu, -12.0, 12.0) / 12.0,
        ],
        dtype=np.float32,
    )
    return values


def mask_feature(mask: np.ndarray) -> np.ndarray:
    patch = normalized_patch(mask)
    return np.concatenate([patch.ravel(), geometric_features(mask)]).astype(np.float32)


def synth_mask(label: str, rng: np.random.Generator, size: int = 128) -> np.ndarray:
    """Generate domain-randomized silhouettes resembling top-view RGB-D masks."""
    image = np.zeros((size, size), np.uint8)
    center = np.array([size / 2, size / 2], dtype=float) + rng.normal(0, 5, 2)
    radius = rng.uniform(19, 39)
    angle = rng.uniform(0, 2 * np.pi)
    if label == "circle":
        theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        points = np.c_[np.cos(theta), np.sin(theta)] * radius
    else:
        n = 4 if label == "square" else 3
        theta = angle + 2 * np.pi * np.arange(n) / n
        points = np.c_[np.cos(theta), np.sin(theta)] * radius
    affine = np.array(
        [
            [rng.uniform(0.82, 1.18), rng.uniform(-0.16, 0.16)],
            [rng.uniform(-0.16, 0.16), rng.uniform(0.82, 1.18)],
        ]
    )
    points = points @ affine.T + center
    cv2.fillPoly(image, [np.rint(points).astype(np.int32)], 255, cv2.LINE_AA)
    image = (image > 127).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    if rng.random() < 0.65:
        image = cv2.erode(image, kernel) if rng.random() < 0.5 else cv2.dilate(image, kernel)
    if rng.random() < 0.35:  # small edge bite / partial depth dropout
        cx, cy = np.rint(center + rng.normal(0, radius * 0.65, 2)).astype(int)
        cv2.circle(image, (cx, cy), int(rng.uniform(2, 6)), 0, -1)
    if rng.random() < 0.25:
        noise = rng.random(image.shape) < rng.uniform(0.0005, 0.002)
        image[noise] = 255 - image[noise]
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image


def make_dataset(n_per_class: int = 1400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x, y = [], []
    for class_index, label in enumerate(CLASSES):
        for _ in range(n_per_class):
            x.append(mask_feature(synth_mask(label, rng)))
            y.append(class_index)
    return np.asarray(x, np.float32), np.asarray(y, np.int64)


def train(
    path: str | Path = "models/shape_mlp_advanced.joblib",
    n_per_class: int = 1400,
    seed: int = 7,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x, y = make_dataset(n_per_class=n_per_class, seed=seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.22, random_state=seed, stratify=y
    )
    classifier = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=8e-4,
        max_iter=180,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=12,
        random_state=seed,
    ).fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    report = classification_report(
        y_test, prediction, target_names=CLASSES, output_dict=True, digits=4, zero_division=0
    )
    bundle: dict[str, Any] = {
        "version": MODEL_VERSION,
        "classes": CLASSES,
        "classifier": classifier,
        "feature_dim": int(x.shape[1]),
        "training": {"seed": seed, "n_per_class": n_per_class, "test_accuracy": report["accuracy"]},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle, report


def load_model(path: str | Path) -> dict[str, Any]:
    loaded = joblib.load(path)
    if isinstance(loaded, dict) and "classifier" in loaded:
        return loaded
    raise ValueError("advanced sorter requires a v2 model bundle; retrain with: python shape_ai.py --train")


def _variants(mask: np.ndarray) -> list[np.ndarray]:
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    return [
        m,
        cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel),
        cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel),
    ]


def predict_distribution(bundle: dict[str, Any], mask: np.ndarray, tta: bool = True) -> np.ndarray:
    classifier = bundle["classifier"]
    variants = _variants(mask) if tta else [(np.asarray(mask) > 0).astype(np.uint8) * 255]
    probabilities = np.asarray([classifier.predict_proba([mask_feature(v)])[0] for v in variants])
    mean = probabilities.mean(axis=0)
    return mean / mean.sum()


def classify(
    bundle: dict[str, Any], mask: np.ndarray, tta: bool = True
) -> tuple[str, float, dict[str, float]]:
    probability = predict_distribution(bundle, mask, tta=tta)
    order = np.argsort(probability)[::-1]
    confidence = float(probability[order[0]])
    margin = float(probability[order[0]] - probability[order[1]])
    entropy = float(-np.sum(probability * np.log(probability + 1e-12)) / np.log(len(CLASSES)))
    label = str(bundle.get("classes", CLASSES)[order[0]])
    return label, confidence, {"margin": margin, "entropy": entropy}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="train and save the advanced MLP")
    parser.add_argument("--model", default="models/shape_mlp_advanced.joblib")
    parser.add_argument("--n-per-class", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not args.train:
        parser.error("specify --train")
    _, report = train(args.model, args.n_per_class, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
