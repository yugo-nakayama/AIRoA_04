"""End-to-end offline validation without Genesis.

Creates a deterministic synthetic RGB-D scene, performs segmentation,
uncertainty-aware classification and 3-D localization, then writes an annotated
image and JSON result under out/offline/.  The model is trained automatically if
it does not exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from perception import CameraIntrinsics, PinholeCamera, RGBDPerception, annotate, look_at_transform
from shape_ai import CLASSES, load_model, synth_mask, train
from supervisor import SelectionPolicy


def make_scene(seed: int = 23, width: int = 640, height: int = 480):
    rng = np.random.default_rng(seed)
    rgb = np.full((height, width, 3), [72, 78, 82], np.uint8)
    depth = np.full((height, width), 1.30, np.float32)
    truth = []
    centers = [(225, 190), (330, 280), (430, 190)]
    colors = [(220, 80, 70), (70, 205, 110), (80, 120, 225)]
    for label, center, color in zip(CLASSES, centers, colors):
        local = synth_mask(label, rng, size=128)
        y0, x0 = center[1] - 64, center[0] - 64
        region = local > 0
        view_depth = depth[y0 : y0 + 128, x0 : x0 + 128]
        view_rgb = rgb[y0 : y0 + 128, x0 : x0 + 128]
        object_height = {"circle": 0.038, "square": 0.044, "triangle": 0.041}[label]
        view_depth[region] = 1.30 - object_height + rng.normal(0, 0.0007, int(region.sum()))
        view_rgb[region] = color
        truth.append({"label": label, "center_uv": center, "height_m": object_height})
    invalid = rng.random(depth.shape) < 0.0004
    depth[invalid] = np.nan
    return rgb, depth, truth


def run(model_path: Path, output_dir: Path, quick_train: bool = False) -> dict:
    if not model_path.exists():
        train(model_path, n_per_class=350 if quick_train else 1400, seed=7)
    model = load_model(model_path)
    rgb, depth, truth = make_scene()
    intrinsics = CameraIntrinsics.from_vertical_fov(640, 480, 50.0)
    transform = look_at_transform((0.55, 0.0, 1.30), (0.55, 0.0, 0.0), (1.0, 0.0, 0.0))
    detector = RGBDPerception(PinholeCamera(intrinsics, transform), model)
    detections = detector.detect(rgb, depth, table_depth_m=1.30)
    policy = SelectionPolicy()
    selected = policy.select(detections)

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = annotate(rgb, detections, None if selected is None else selected.detection_id)
    cv2.imwrite(str(output_dir / "offline_rgbd_result.png"), overlay)
    result = {
        "truth": truth,
        "detections": [d.record() for d in detections],
        "selected_id": None if selected is None else selected.detection_id,
        "detected_labels": sorted(d.label for d in detections),
        "pass": len(detections) == 3 and sorted(d.label for d in detections) == sorted(CLASSES),
    }
    (output_dir / "offline_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/shape_mlp_advanced.joblib"))
    parser.add_argument("--out", type=Path, default=Path("out/offline"))
    parser.add_argument("--quick-train", action="store_true")
    args = parser.parse_args()
    result = run(args.model, args.out, args.quick_train)
    print(json.dumps({k: result[k] for k in ("detected_labels", "selected_id", "pass")}, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
