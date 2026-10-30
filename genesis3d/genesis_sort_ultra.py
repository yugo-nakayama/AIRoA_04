"""Ultra Genesis sorter: Franka Panda + fixed overhead RGB-D camera.

Compared with the baseline, this executable adds:
- calibrated pinhole back-projection and robust table-depth estimation;
- uncertainty-aware MLP classification with explicit rejection;
- closed-loop re-observation immediately before descent;
- two-signal grasp verification (finger width + visual removal from table);
- failure-specific retries and machine-readable JSONL telemetry.

Genesis itself is imported only inside main(), so all non-physics modules remain
unit-testable on machines where genesis-world is not installed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception import (  # noqa: E402
    CameraIntrinsics,
    PinholeCamera,
    RGBDPerception,
    annotate,
    associate_nearest,
    look_at_transform,
)
from shape_ai import load_model  # noqa: E402
from adaptive_grasp import SuccessPredictor, append_experience, generate_candidates  # noqa: E402
from decision import Action, decide  # noqa: E402
from novelty import GeometryNoveltyDetector  # noqa: E402
from supervisor import (  # noqa: E402
    EpisodeState,
    Failure,
    Phase,
    RecoveryPolicy,
    SelectionPolicy,
    VerificationPolicy,
)
from telemetry import Telemetry, write_summary  # noqa: E402


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    return array[0] if array.ndim == 4 else array


def render_camera(camera: Any, with_depth: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    output = camera.render(rgb=True, depth=with_depth)
    rgb = to_numpy(output[0]).astype(np.uint8)
    if not with_depth:
        return rgb, None
    depth = to_numpy(output[1]).astype(np.float32).squeeze()
    return rgb, depth


def quaternion_down(yaw: float) -> np.ndarray:
    """Genesis quaternion in wxyz; gripper z-axis points downward."""
    return np.array([0.0, np.cos(yaw / 2.0), np.sin(yaw / 2.0), 0.0])


def quaternion_z(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def write_triangular_prism(path: Path, side: float = 0.05, height: float = 0.04) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    radius = side / np.sqrt(3.0)
    vertices = [(radius * np.cos(a), radius * np.sin(a)) for a in np.deg2rad([90, 210, 330])]
    with path.open("w", encoding="ascii") as stream:
        for z in (-height / 2.0, height / 2.0):
            for x, y in vertices:
                stream.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for face in ("1 3 2", "4 5 6", "1 2 5 4", "2 3 6 5", "3 1 4 6"):
            stream.write(f"f {face}\n")


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp_force(base_force: float, scale: float, limit_force: float) -> float:
    """Both configured forces are negative closing commands."""
    requested = base_force * scale
    return float(max(requested, limit_force))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ultra.json")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "shape_mlp_advanced.joblib")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--grasp-model", type=Path, default=ROOT / "out" / "experiments" / "models" / "grasp_success_hgb.joblib")
    parser.add_argument("--novelty-model", type=Path, default=ROOT / "out" / "experiments" / "models" / "geometry_novelty.joblib")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    try:
        import genesis as gs
    except ImportError as exc:
        raise SystemExit(
            "genesis-world is not installed. Run the offline tests first, then install the Genesis extra."
        ) from exc

    config = load_config(args.config)
    seed = int(config["seed"] if args.seed is None else args.seed)
    rng = np.random.default_rng(seed)
    model = load_model(args.model)
    grasp_model = SuccessPredictor.load(args.grasp_model)
    novelty_model = GeometryNoveltyDetector.load(args.novelty_model)
    out_dir = ROOT / "out" / "genesis"
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = ROOT / "assets"
    triangle_path = assets_dir / "tri_prism.obj"
    write_triangular_prism(triangle_path, height=config["workspace"]["object_height_m"])

    camera_cfg = config["camera"]
    workspace_cfg = config["workspace"]
    perception_cfg = config["perception"]
    grasp_cfg = config["grasp"]
    bins = {key: tuple(value) for key, value in config["bins"].items()}
    resolution = tuple(camera_cfg["resolution"])
    object_height = float(workspace_cfg["object_height_m"])

    selection = SelectionPolicy(
        min_confidence=float(perception_cfg["min_confidence"]),
        min_margin=float(perception_cfg["min_margin"]),
        max_entropy=float(perception_cfg["max_entropy"]),
        max_depth_mad_m=float(perception_cfg["max_depth_mad_m"]),
        workspace_x=tuple(workspace_cfg["x"]),
        workspace_y=tuple(workspace_cfg["y"]),
    )
    verification = VerificationPolicy(
        min_finger_width_m=float(grasp_cfg["finger_object_threshold_m"]),
        min_lift_height_delta_m=float(grasp_cfg["required_lift_delta_m"]),
    )
    recovery = RecoveryPolicy(max_attempts_per_object=int(grasp_cfg["max_attempts_per_object"]))
    telemetry = Telemetry(
        out_dir / "events.jsonl",
        {"seed": seed, "config": str(args.config), "model": str(args.model), "episodes": args.episodes},
    )

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, seed=seed)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.6, -1.2, 1.2), camera_lookat=(0.5, -0.1, 0.1), camera_fov=40
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane())
    bin_colors = {
        "circle": (0.85, 0.30, 0.30, 1.0),
        "square": (0.30, 0.75, 0.30, 1.0),
        "triangle": (0.30, 0.45, 0.90, 1.0),
    }
    for label, (bx, by) in bins.items():
        scene.add_entity(
            gs.morphs.Box(size=(0.15, 0.15, 0.002), pos=(bx, by, 0.001), fixed=True),
            surface=gs.surfaces.Default(color=bin_colors[label]),
        )

    object_labels = ["circle", "square", "triangle"] * 2
    objects: list[dict[str, Any]] = []
    for index, label in enumerate(object_labels):
        color = tuple(float(v) for v in rng.uniform(0.15, 0.85, 3)) + (1.0,)
        parked = (0.18, 0.70 + 0.08 * index, object_height / 2.0)
        if label == "circle":
            morph = gs.morphs.Cylinder(radius=0.022, height=object_height, pos=parked)
        elif label == "square":
            morph = gs.morphs.Box(size=(0.04, 0.04, object_height), pos=parked)
        else:
            morph = gs.morphs.Mesh(file=str(triangle_path.resolve()), pos=parked)
        entity = scene.add_entity(morph, surface=gs.surfaces.Default(color=color))
        objects.append({"id": index, "label": label, "entity": entity})

    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    overhead = scene.add_camera(
        res=resolution,
        pos=tuple(camera_cfg["position"]),
        lookat=tuple(camera_cfg["look_at"]),
        up=tuple(camera_cfg["up"]),
        fov=float(camera_cfg["vertical_fov_deg"]),
        GUI=False,
    )
    observer = scene.add_camera(
        res=resolution,
        pos=(1.45, -1.05, 0.95),
        lookat=(0.5, -0.12, 0.08),
        fov=40,
        GUI=False,
    )
    scene.build()

    motors = np.arange(7)
    fingers = np.arange(7, 9)
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )
    end_effector = franka.get_link("hand")

    intrinsics = CameraIntrinsics.from_vertical_fov(
        resolution[0], resolution[1], float(camera_cfg["vertical_fov_deg"])
    )
    transform = look_at_transform(camera_cfg["position"], camera_cfg["look_at"], camera_cfg["up"])
    camera_model = PinholeCamera(intrinsics, transform)
    perception = RGBDPerception(
        camera_model,
        model,
        min_height_m=float(perception_cfg["min_height_m"]),
        max_height_m=float(perception_cfg["max_height_m"]),
        min_area_px=int(perception_cfg["min_area_px"]),
    )

    record_every = int(config["recording"]["record_every_steps"])
    frame_size = (resolution[0] * 2, resolution[1] + 68)
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(out_dir / "genesis_raw.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            int(config["recording"]["fps"]),
            frame_size,
        )
    visual_state: dict[str, Any] = {"step": 0, "status": "initializing", "detections": [], "selected": None}

    def write_frame() -> None:
        if writer is None:
            return
        side_rgb, _ = render_camera(observer)
        top_rgb, _ = render_camera(overhead)
        selected_id = None if visual_state["selected"] is None else visual_state["selected"].detection_id
        top_bgr = annotate(top_rgb, visual_state["detections"], selected_id)
        side_bgr = cv2.cvtColor(side_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(side_bgr, "Genesis / Franka Panda", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(top_bgr, "RGB-D + uncertainty-aware MLP", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        bar = np.full((68, frame_size[0], 3), 36, np.uint8)
        cv2.putText(bar, visual_state["status"], (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2)
        writer.write(np.vstack([np.hstack([side_bgr, top_bgr]), bar]))

    def step(count: int = 1) -> None:
        for _ in range(count):
            scene.step()
            visual_state["step"] += 1
            if visual_state["step"] % record_every == 0:
                write_frame()

    def move_to(position: tuple[float, float, float], yaw: float, steps: int = 75, settle: int = 20) -> bool:
        try:
            q_goal = to_numpy(
                franka.inverse_kinematics(
                    link=end_effector, pos=np.asarray(position), quat=quaternion_down(yaw)
                )
            ).ravel()[:7]
            if not np.isfinite(q_goal).all():
                return False
            q_start = to_numpy(franka.get_dofs_position()).ravel()[:7]
            for index in range(steps):
                fraction = (index + 1) / steps
                smooth = 3 * fraction**2 - 2 * fraction**3
                franka.control_dofs_position(q_start + (q_goal - q_start) * smooth, motors)
                step()
            step(settle)
            return True
        except Exception as exc:  # Genesis versions differ in IK failure behavior
            telemetry.emit("ik_error", error=repr(exc), position=position, yaw=yaw)
            return False

    def open_gripper() -> None:
        franka.control_dofs_position(np.array([0.04, 0.04]), fingers)
        step(35)

    def close_gripper(force_scale: float, preclose_width_m: float = 0.050, learned_force_n: float | None = None) -> float:
        force = clamp_force(
            -float(learned_force_n) if learned_force_n is not None else float(grasp_cfg["base_force_n"]),
            force_scale, float(grasp_cfg["force_limit_n"])
        )
        # Pre-close in position mode prevents an impulsive full-force contact.
        half_width = float(np.clip(preclose_width_m / 2.0, 0.018, 0.035))
        franka.control_dofs_position(np.array([half_width, half_width]), fingers)
        step(20)
        franka.control_dofs_force(np.array([force, force]), fingers)
        step(35)
        return force

    def observe() -> tuple[np.ndarray, np.ndarray, list[Any]]:
        rgb, depth = render_camera(overhead, with_depth=True)
        assert depth is not None
        detections = perception.detect(rgb, depth)
        visual_state["detections"] = detections
        telemetry.emit("observation", detections=[d.record() for d in detections])
        return rgb, depth, detections

    def adaptive_observation(detection: Any) -> dict[str, Any]:
        """Convert RGB-D output to the features used by the grasp and OOD models."""
        record: dict[str, Any] = {
            "predicted_label": detection.label,
            "confidence": detection.confidence,
            "margin": detection.margin,
            "entropy": detection.entropy,
            "height_m": detection.height_m,
            "depth_mad_m": detection.depth_mad_m,
            "edge_clearance_m": 0.05,
            # Conservative online estimates; replace with material estimator when available.
            "friction_estimate": 0.55,
            "position_sigma_m": max(0.001, detection.depth_mad_m * 1.8),
            "orientation_sigma_rad": 0.12,
            "spacing_m": 0.08,
            **detection.geometry,
        }
        record.update(novelty_model.evaluate(record, detection.label, detection.confidence))
        return record

    def place_objects() -> None:
        positions: list[np.ndarray] = []
        for item in objects:
            for _ in range(250):
                candidate = np.array(
                    [rng.uniform(*workspace_cfg["x"]), rng.uniform(*workspace_cfg["y"])]
                )
                if all(np.linalg.norm(candidate - existing) > 0.10 for existing in positions):
                    break
            positions.append(candidate)
            item["entity"].set_pos(np.array([*candidate, object_height / 2.0 + 0.002]))
            item["entity"].set_quat(quaternion_z(rng.uniform(0.0, 2.0 * np.pi)))
        step(35)

    def nearest_truth(xy: np.ndarray) -> dict[str, Any]:
        return min(
            objects,
            key=lambda item: np.linalg.norm(to_numpy(item["entity"].get_pos()).ravel()[:2] - xy),
        )

    home = (0.47, 0.0, float(workspace_cfg["safe_z"]) + 0.12)
    open_gripper()
    if not move_to(home, 0.0, steps=65):
        raise RuntimeError("home pose IK failed")

    for episode in range(args.episodes):
        episode_state = EpisodeState()
        place_objects()
        telemetry.emit("episode_start", episode=episode + 1)
        completed_ids: set[int] = set()
        pending_plans: dict[str, Any] = {}
        deferred_cycles = 0

        while len(completed_ids) < len(objects) and deferred_cycles < 8:
            episode_state.transition(Phase.OBSERVE)
            visual_state["status"] = f"Episode {episode + 1}: observe / select"
            _, _, detections = observe()
            # Ignore detections that map to already finished objects.
            active = []
            for det in detections:
                mapped = nearest_truth(det.position_world[:2])
                if mapped["id"] not in completed_ids:
                    active.append(det)
            target = selection.select(active)
            visual_state["selected"] = target
            if target is None:
                uncertain = [d.record() for d in detections if d.usable]
                telemetry.emit("selection_rejected", episode=episode + 1, candidates=uncertain)
                deferred_cycles += 1
                step(30)
                continue

            # Open-set rejection is based on class-conditional geometry support, not
            # closed-set MLP confidence alone.  Unknowns are left on the table.
            adaptive_obs = adaptive_observation(target)
            candidates = generate_candidates(adaptive_obs, 0)
            adaptive_candidate, candidate_probabilities = grasp_model.select(adaptive_obs, candidates, 0)
            initial_decision = decide(
                adaptive_obs,
                float(np.max(candidate_probabilities)),
                reobservations=0,
                enable_reobserve=True,
                enable_unknown_hold=True,
            )
            if initial_decision.action == Action.HOLD:
                telemetry.emit(
                    "adaptive_hold", episode=episode + 1, detection=target.record(),
                    observation=adaptive_obs, reason=initial_decision.reason,
                )
                deferred_cycles += 1
                step(25)
                continue
            if initial_decision.action == Action.REOBSERVE:
                # Agreed first implementation: move the arm fully clear, then reacquire
                # from the same fixed overhead camera.  Camera viewpoint change is out of scope.
                visual_state["status"] = f"{target.label}: arm-clear re-observation"
                move_to(home, 0.0, steps=45)
                step(18)
                _, _, reacquired = observe()
                updated = associate_nearest(target, reacquired)
                if updated is None:
                    telemetry.emit("adaptive_hold", episode=episode + 1, detection=target.record(), reason="target lost during arm-clear re-observation")
                    deferred_cycles += 1
                    continue
                target = updated
                adaptive_obs = adaptive_observation(target)
                candidates = generate_candidates(adaptive_obs, 0)
                adaptive_candidate, candidate_probabilities = grasp_model.select(adaptive_obs, candidates, 0)
                telemetry.emit(
                    "adaptive_reobservation", episode=episode + 1,
                    detection=target.record(), predicted_success=float(np.max(candidate_probabilities)),
                )
            truth = nearest_truth(target.position_world[:2])
            key = f"ep{episode + 1}-object{truth['id']}"
            if truth["id"] in completed_ids:
                # A bin or already-placed object entered the workspace mask; ignore it.
                telemetry.emit("duplicate_target", key=key)
                deferred_cycles += 1
                step(15)
                continue

            deferred_cycles = 0  # only reset when starting work on a fresh object
            attempt = episode_state.attempt(key)
            failure = Failure.NONE
            plan = pending_plans.pop(key, recovery.plan(Failure.NONE, attempt))
            # Regenerate for the actual attempt so failure history changes the candidate set.
            candidates = generate_candidates(adaptive_obs, attempt)
            adaptive_candidate, candidate_probabilities = grasp_model.select(adaptive_obs, candidates, attempt)
            xy = target.position_world[:2] + np.asarray(plan.xy_offset_m) + np.asarray([adaptive_candidate.dx_m, adaptive_candidate.dy_m])
            yaw = target.yaw_world + plan.yaw_offset_rad + adaptive_candidate.yaw_offset_rad
            force_scale = plan.force_scale
            telemetry.emit(
                "grasp_candidate_selected", key=key, attempt=attempt + 1,
                candidate=adaptive_candidate.record(),
                candidate_probabilities={c.candidate_id: float(p) for c, p in zip(candidates, candidate_probabilities)},
            )
            visual_state["status"] = f"{target.label}: approach (confidence {target.confidence:.2f})"
            episode_state.transition(Phase.APPROACH)
            if not move_to((float(xy[0]), float(xy[1]), float(workspace_cfg["approach_z"])), yaw):
                failure = Failure.IK_FAILED
            else:
                # Reobserve after the arm moved: compensate target motion before descent.
                # Overhead camera is often occluded by the arm; treat lost/large jumps as soft.
                episode_state.transition(Phase.REALIGN)
                _, _, current = observe()
                updated = associate_nearest(target, current, max_distance_m=0.08)
                if updated is None:
                    telemetry.emit("realign_soft_fail", key=key, reason="target_occluded_keep_prior")
                else:
                    correction = updated.position_world[:2] - target.position_world[:2]
                    corr_n = float(np.linalg.norm(correction))
                    max_corr = float(perception_cfg["closed_loop_max_correction_m"])
                    if corr_n <= max_corr:
                        xy = updated.position_world[:2] + np.asarray(plan.xy_offset_m) + np.asarray(
                            [adaptive_candidate.dx_m, adaptive_candidate.dy_m]
                        )
                        target = updated
                    else:
                        telemetry.emit(
                            "realign_soft_fail",
                            key=key,
                            reason="correction_too_large_keep_prior",
                            correction_m=corr_n,
                        )

            width_at_grasp = 0.0
            visually_removed = False
            if failure == Failure.NONE:
                episode_state.transition(Phase.DESCEND)
                z_center = target.position_world[2] - object_height / 2.0
                grasp_z = z_center + float(grasp_cfg["hand_down_offset_m"])
                if not move_to((float(xy[0]), float(xy[1]), float(grasp_z)), yaw, steps=50):
                    failure = Failure.IK_FAILED
                else:
                    episode_state.transition(Phase.GRASP)
                    force = close_gripper(
                        force_scale,
                        preclose_width_m=adaptive_candidate.width_m,
                        learned_force_n=adaptive_candidate.force_n,
                    )
                    telemetry.emit("grasp_force", key=key, force_n=force)
                    if not move_to((float(xy[0]), float(xy[1]), 0.29), yaw, steps=55):
                        failure = Failure.IK_FAILED
                    else:
                        width_at_grasp = float(
                            np.sum(to_numpy(franka.get_dofs_position()).ravel()[7:9])
                        )
                        # At table-height segmentation, a successfully lifted target disappears.
                        _, _, after = observe()
                        still_on_table = associate_nearest(target, after, max_distance_m=0.055)
                        visually_removed = still_on_table is None
                        episode_state.transition(Phase.VERIFY)
                        fingers_ok = verification.finger_has_object(width_at_grasp)
                        if fingers_ok and visually_removed:
                            telemetry.emit(
                                "verify_ok",
                                key=key,
                                finger_width_m=width_at_grasp,
                                visual_table_clear=True,
                            )
                        else:
                            pseudo_after_height = (
                                target.height_m + verification.min_lift_height_delta_m
                                if visually_removed
                                else (still_on_table.height_m if still_on_table is not None else None)
                            )
                            ok = verification.success(
                                width_at_grasp, float(target.height_m), pseudo_after_height
                            )
                            telemetry.emit(
                                "verify_detail",
                                key=key,
                                finger_width_m=width_at_grasp,
                                fingers_ok=fingers_ok,
                                visual_table_clear=visually_removed,
                                ok=ok,
                            )
                            if not ok:
                                failure = Failure.EMPTY_GRASP

            success = False
            placed_correctly = False
            if failure == Failure.NONE:
                episode_state.transition(Phase.TRANSPORT)
                bx, by = bins[target.label]
                visual_state["status"] = f"{target.label}: transport / place"
                if not move_to((bx, by, 0.29), yaw, steps=90):
                    failure = Failure.IK_FAILED
                else:
                    width_after_motion = float(
                        np.sum(to_numpy(franka.get_dofs_position()).ravel()[7:9])
                    )
                    if verification.slipped(width_at_grasp, width_after_motion):
                        failure = Failure.SLIP
                    elif not move_to((bx, by, 0.18), yaw, steps=42):
                        failure = Failure.IK_FAILED
                    else:
                        episode_state.transition(Phase.PLACE)
                        open_gripper()
                        move_to((bx, by, 0.31), yaw, steps=42)
                        final_position = to_numpy(truth["entity"].get_pos()).ravel()
                        placed_correctly = bool(
                            np.linalg.norm(final_position[:2] - np.asarray(bins[truth["label"]])) < 0.09
                        )
                        success = placed_correctly
                        failure = Failure.NONE if success else Failure.SLIP

            attempts_used = episode_state.object_attempts[key]
            if success:
                pending_plans.pop(key, None)
                completed_ids.add(truth["id"])
                visual_state["status"] = f"success: {truth['label']}"
                telemetry.emit(
                    "object_result",
                    episode=episode + 1,
                    key=key,
                    success=True,
                    failure=Failure.NONE.value,
                    attempts=attempts_used,
                    predicted_label=target.label,
                    true_label=truth["label"],
                    confidence=target.confidence,
                    finger_width_m=width_at_grasp,
                    visual_table_clear=visually_removed,
                    placed_correctly=placed_correctly,
                )
                append_experience(out_dir / "grasp_experience.jsonl", adaptive_obs, adaptive_candidate, attempt, True, Failure.NONE.value, key)
            else:
                next_plan = recovery.plan(failure, attempts_used)
                visual_state["status"] = f"recover: {failure.value} / {next_plan.reason}"
                telemetry.emit(
                    "object_result",
                    episode=episode + 1,
                    key=key,
                    success=False,
                    failure=failure.value,
                    attempts=attempts_used,
                    predicted_label=target.label,
                    true_label=truth["label"],
                    confidence=target.confidence,
                    finger_width_m=width_at_grasp,
                    visual_table_clear=visually_removed,
                    recovery=next_plan.__dict__,
                )
                append_experience(out_dir / "grasp_experience.jsonl", adaptive_obs, adaptive_candidate, attempt, False, failure.value, key)
                open_gripper()
                move_to((float(xy[0]), float(xy[1]), next_plan.retreat_height_m), yaw, steps=45)
                if next_plan.retry:
                    pending_plans[key] = next_plan
                else:
                    pending_plans.pop(key, None)
                    completed_ids.add(truth["id"])
            episode_state.transition(Phase.RETREAT)
            move_to(home, 0.0, steps=65)
            visual_state["selected"] = None

        telemetry.emit(
            "episode_end",
            episode=episode + 1,
            completed=len(completed_ids),
            total=len(objects),
            transitions=episode_state.transitions,
        )
        move_to(home, 0.0, steps=60)

    if writer is not None:
        writer.release()
    summary = write_summary(
        out_dir / "summary.json",
        telemetry.records,
        genesis_executed=True,
        seed=seed,
        total_expected_objects=len(objects) * args.episodes,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
