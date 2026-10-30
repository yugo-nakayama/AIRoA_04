"""RGB-D perception and 3-D localization for the advanced sorter.

Coordinate convention follows OpenCV: camera +x right, +y down, +z forward.
T_world_cam maps homogeneous camera points to world coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import cv2
import numpy as np

from shape_ai import classify, geometric_features


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_vertical_fov(cls, width: int, height: int, fov_deg: float) -> "CameraIntrinsics":
        fy = (height / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
        return cls(float(fy), float(fy), width / 2.0, height / 2.0, width, height)


@dataclass
class Detection:
    detection_id: int
    label: str
    confidence: float
    margin: float
    entropy: float
    centroid_uv: tuple[float, float]
    position_world: np.ndarray
    yaw_world: float
    height_m: float
    area_px: int
    depth_mad_m: float
    touching_border: bool
    mask: np.ndarray = field(repr=False)
    contour: np.ndarray = field(repr=False)
    geometry: dict[str, float] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return not self.touching_border and np.isfinite(self.position_world).all()

    def record(self) -> dict[str, Any]:
        return {
            "id": self.detection_id,
            "label": self.label,
            "confidence": round(self.confidence, 5),
            "margin": round(self.margin, 5),
            "entropy": round(self.entropy, 5),
            "uv": [round(x, 3) for x in self.centroid_uv],
            "xyz_m": [round(float(x), 5) for x in self.position_world],
            "yaw_rad": round(self.yaw_world, 5),
            "height_m": round(self.height_m, 5),
            "area_px": self.area_px,
            "depth_mad_m": round(self.depth_mad_m, 6),
            "touching_border": self.touching_border,
            "geometry": {k: round(float(v), 6) for k, v in self.geometry.items()},
        }


def look_at_transform(
    position: Iterable[float], target: Iterable[float], up_world: Iterable[float]
) -> np.ndarray:
    """Construct T_world_cam for an OpenCV-style pinhole camera."""
    p = np.asarray(position, float)
    target = np.asarray(target, float)
    up = np.asarray(up_world, float)
    z_forward = target - p
    z_forward /= np.linalg.norm(z_forward)
    x_right = np.cross(z_forward, up)
    x_right /= np.linalg.norm(x_right)
    y_down = np.cross(z_forward, x_right)
    y_down /= np.linalg.norm(y_down)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.column_stack([x_right, y_down, z_forward])
    transform[:3, 3] = p
    return transform


class PinholeCamera:
    def __init__(self, intrinsics: CameraIntrinsics, t_world_cam: np.ndarray):
        self.k = intrinsics
        self.t_world_cam = np.asarray(t_world_cam, float)
        if self.t_world_cam.shape != (4, 4):
            raise ValueError("T_world_cam must be 4x4")
        self.t_cam_world = np.linalg.inv(self.t_world_cam)

    def backproject(self, u: np.ndarray | float, v: np.ndarray | float, depth: np.ndarray | float) -> np.ndarray:
        u, v, depth = np.broadcast_arrays(u, v, depth)
        x = (u - self.k.cx) * depth / self.k.fx
        y = (v - self.k.cy) * depth / self.k.fy
        camera = np.stack([x, y, depth, np.ones_like(depth)], axis=-1)
        return camera @ self.t_world_cam.T[..., :]

    def project(self, xyz_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xyz = np.asarray(xyz_world, float)
        hom = np.concatenate([xyz, np.ones((*xyz.shape[:-1], 1))], axis=-1)
        camera = hom @ self.t_cam_world.T
        z = camera[..., 2]
        u = self.k.fx * camera[..., 0] / z + self.k.cx
        v = self.k.fy * camera[..., 1] / z + self.k.cy
        return np.stack([u, v], axis=-1), z


def robust_table_depth(depth: np.ndarray, valid_range: tuple[float, float] = (0.2, 3.0)) -> float:
    """Estimate flat table depth from the dominant far-depth band.

    Top-view objects are closer than the table.  Taking the median of the upper
    depth quintile is stable even when objects occupy a sizeable central region.
    """
    d = np.asarray(depth, np.float32)
    valid = d[(d > valid_range[0]) & (d < valid_range[1]) & np.isfinite(d)]
    if valid.size < 100:
        raise ValueError("not enough valid depth pixels")
    cutoff = np.quantile(valid, 0.72)
    return float(np.median(valid[valid >= cutoff]))


def _world_yaw(camera: PinholeCamera, contour: np.ndarray, depth_value: float) -> float:
    (_, _), (w, h), angle_deg = cv2.minAreaRect(contour)
    angle = np.deg2rad(angle_deg + (90.0 if w < h else 0.0))
    direction = np.array([np.cos(angle), np.sin(angle)]) * 12.0
    center = contour.reshape(-1, 2).mean(axis=0)
    p = camera.backproject(
        np.array([center[0], center[0] + direction[0]]),
        np.array([center[1], center[1] + direction[1]]),
        np.array([depth_value, depth_value]),
    )[:, :3]
    delta = p[1] - p[0]
    return float(np.arctan2(delta[1], delta[0]))


class RGBDPerception:
    def __init__(
        self,
        camera: PinholeCamera,
        model_bundle: dict[str, Any],
        min_height_m: float = 0.012,
        max_height_m: float = 0.095,
        min_area_px: int = 70,
        border_px: int = 2,
    ):
        self.camera = camera
        self.model = model_bundle
        self.min_height_m = min_height_m
        self.max_height_m = max_height_m
        self.min_area_px = min_area_px
        self.border_px = border_px

    def segment(self, depth: np.ndarray, table_depth_m: float | None = None) -> tuple[np.ndarray, float]:
        depth = np.asarray(depth, np.float32)
        table_depth_m = robust_table_depth(depth) if table_depth_m is None else float(table_depth_m)
        height = table_depth_m - depth
        valid = np.isfinite(depth) & (height > self.min_height_m) & (height < self.max_height_m)
        mask = valid.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask, table_depth_m

    def detect(self, rgb: np.ndarray, depth: np.ndarray, table_depth_m: float | None = None) -> list[Detection]:
        del rgb  # retained in the interface for future RGB/depth fusion
        depth = np.asarray(depth, np.float32)
        mask, table_depth_m = self.segment(depth, table_depth_m)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        detections: list[Detection] = []
        next_id = 0
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < self.min_area_px:
                continue
            component_mask = (labels == component).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            pixels = labels == component
            component_depth = depth[pixels]
            finite = component_depth[np.isfinite(component_depth)]
            if finite.size < self.min_area_px:
                continue
            depth_median = float(np.median(finite))
            depth_mad = float(np.median(np.abs(finite - depth_median)))
            u, v = map(float, centroids[component])
            world = self.camera.backproject(u, v, depth_median)[:3]
            height = float(table_depth_m - depth_median)
            label, confidence, uncertainty = classify(self.model, component_mask, tta=True)
            gf = geometric_features(component_mask)
            geometry = {
                "circularity": float(gf[0]),
                "solidity": float(gf[1]),
                "aspect": float(gf[3]),
                "vertices_norm": float(gf[4]),
                "edge_damage": float(max(0.0, 1.0 - gf[1])),
            }
            x, y, w, h = stats[component, :4]
            border = (
                x <= self.border_px
                or y <= self.border_px
                or x + w >= depth.shape[1] - self.border_px
                or y + h >= depth.shape[0] - self.border_px
            )
            detections.append(
                Detection(
                    detection_id=next_id,
                    label=label,
                    confidence=confidence,
                    margin=uncertainty["margin"],
                    entropy=uncertainty["entropy"],
                    centroid_uv=(u, v),
                    position_world=np.asarray(world, float),
                    yaw_world=_world_yaw(self.camera, contour, depth_median),
                    height_m=height,
                    area_px=area,
                    depth_mad_m=depth_mad,
                    touching_border=bool(border),
                    mask=component_mask,
                    contour=contour,
                    geometry=geometry,
                )
            )
            next_id += 1
        return detections


def associate_nearest(
    previous: Detection, current: list[Detection], max_distance_m: float = 0.06
) -> Detection | None:
    if not current:
        return None
    same_label = [d for d in current if d.label == previous.label and d.usable]
    candidates = same_label or [d for d in current if d.usable]
    if not candidates:
        return None
    candidate = min(candidates, key=lambda d: np.linalg.norm(d.position_world[:2] - previous.position_world[:2]))
    distance = float(np.linalg.norm(candidate.position_world[:2] - previous.position_world[:2]))
    return candidate if distance <= max_distance_m else None


def annotate(rgb: np.ndarray, detections: list[Detection], selected_id: int | None = None) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    for detection in detections:
        selected = detection.detection_id == selected_id
        color = (40, 40, 240) if selected else ((40, 190, 40) if detection.usable else (0, 170, 255))
        cv2.drawContours(canvas, [detection.contour], -1, color, 2)
        u, v = map(int, detection.centroid_uv)
        text = f"{detection.label} p={detection.confidence:.2f} H={detection.entropy:.2f}"
        cv2.putText(canvas, text, (max(2, u - 70), max(18, v - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return canvas
