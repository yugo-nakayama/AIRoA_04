"""Reproducible A/B/C/D, ablation and unseen-condition experiments.

This is a controlled stochastic contact-outcome benchmark.  It executes the real
candidate generator, learned success predictor, open-set detector, action policy,
dual grasp verification and placement checks while using a documented probabilistic
contact model.  Genesis qualification is performed separately by
``genesis3d/qualify_genesis.py``; surrogate results are never labelled as Genesis.
"""
from __future__ import annotations

import argparse, hashlib, json, math, platform, subprocess, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from adaptive_grasp import GraspCandidate, SuccessPredictor, generate_candidates
from decision import Action, decide
from novelty import GeometryNoveltyDetector

ROOT = Path(__file__).resolve().parent
CLASSES = ("circle", "square", "triangle")
UNKNOWN = ("star", "pentagon", "crescent", "damaged")
BASE_GEOM = {
    "circle": np.array([0.91, 0.995, 0.96, 1.28]),
    "square": np.array([0.785, 0.992, 0.97, 0.50]),
    "triangle": np.array([0.60, 0.985, 0.83, 0.375]),
    "star": np.array([0.40, 0.66, 0.72, 1.25]),
    "pentagon": np.array([0.75, 0.985, 0.91, 0.625]),
    "crescent": np.array([0.36, 0.61, 0.58, 1.05]),
    "damaged": np.array([0.67, 0.76, 0.84, 0.72]),
}


@dataclass(frozen=True)
class Condition:
    name: str; pos_mm: float; rot_deg: float; depth_mm: float; friction: float; spacing_mm: float; shapes: tuple[str, ...]; split: str


TRAIN_CONDITION = Condition("train_domain", 6, 10, 2.5, 0.60, 90, CLASSES, "train")
EVAL_CONDITIONS = (
    Condition("nominal", 2, 3, 0.8, 0.65, 110, CLASSES, "eval"),
    Condition("position_shift", 12, 5, 1.5, 0.65, 90, CLASSES, "eval"),
    Condition("rotation_shift", 5, 22, 1.5, 0.65, 90, CLASSES, "eval"),
    Condition("depth_noise", 5, 8, 7.0, 0.65, 90, CLASSES, "eval"),
    Condition("low_friction", 5, 8, 2.0, 0.28, 90, CLASSES, "unseen"),
    Condition("crowded", 7, 12, 3.0, 0.55, 45, CLASSES, "unseen"),
    Condition("combined_unseen", 14, 28, 8.0, 0.25, 42, CLASSES, "unseen"),
    Condition("open_set", 7, 14, 4.0, 0.50, 65, CLASSES + UNKNOWN, "unseen"),
)
CONFIGS = {
    "A": dict(learned=False, reobserve=False, unknown_hold=False, geometry_novelty=False),
    "B": dict(learned=True, reobserve=False, unknown_hold=False, geometry_novelty=False),
    "C": dict(learned=True, reobserve=True, unknown_hold=False, geometry_novelty=False),
    "D": dict(learned=True, reobserve=True, unknown_hold=True, geometry_novelty=True),
}
ABLATIONS = {
    "D_full": CONFIGS["D"],
    "D_no_geometry_novelty": dict(learned=True, reobserve=True, unknown_hold=True, geometry_novelty=False),
    "D_no_reobserve": dict(learned=True, reobserve=False, unknown_hold=True, geometry_novelty=True),
    "D_no_learned_selector": dict(learned=False, reobserve=True, unknown_hold=True, geometry_novelty=True),
    "D_no_TTA_proxy": dict(learned=True, reobserve=True, unknown_hold=True, geometry_novelty=True, no_tta=True),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def observe(shape: str, condition: Condition, rng: np.random.Generator, reobserved: bool = False, no_tta: bool = False) -> dict[str, Any]:
    factor = 0.52 if reobserved else 1.0
    geom_sigma = 0.012 + factor * (condition.depth_mm / 1000 * 2.8 + condition.rot_deg / 1800)
    if no_tta: geom_sigma *= 1.25
    geom = BASE_GEOM[shape] + rng.normal(0, geom_sigma, 4)
    geom[0:3] = np.clip(geom[0:3], 0.05, 1.05); geom[3] = np.clip(geom[3], 0.25, 1.5)
    dists = {k: float(np.linalg.norm((geom - BASE_GEOM[k]) / np.array([0.08, 0.07, 0.10, 0.20]))) for k in CLASSES}
    predicted = min(dists, key=dists.get)
    sorted_d = sorted(dists.values())
    ambiguity = max(0.0, sorted_d[1] - sorted_d[0])
    # Closed-set confidence can remain high on unknowns; geometry novelty is separate.
    confidence = float(np.clip(0.60 + 0.38 * (1 - math.exp(-ambiguity)) - factor * condition.depth_mm / 90 + rng.normal(0, 0.025), 0.34, 0.997))
    margin = float(np.clip(confidence - (1 - confidence) * 0.55, 0.02, 0.995))
    entropy = float(np.clip(1.08 - confidence + rng.normal(0, 0.025), 0.005, 0.95))
    edge_damage = float(np.clip((0.29 if shape == "damaged" else 0.02) + rng.normal(0, 0.035), 0, 0.6))
    return {
        "true_shape": shape, "known": shape in CLASSES, "predicted_label": predicted,
        "confidence": confidence, "margin": margin, "entropy": entropy,
        "circularity": float(geom[0]), "solidity": float(geom[1]), "aspect": float(geom[2]),
        "vertices_norm": float(geom[3]), "edge_damage": edge_damage,
        "height_m": float(np.clip(0.040 + rng.normal(0, condition.depth_mm / 3000), 0.025, 0.060)),
        "depth_mad_m": float(condition.depth_mm / 1000 * rng.uniform(0.55, 1.05) * factor),
        "edge_clearance_m": float(max(condition.spacing_mm / 2000 + rng.normal(0, 0.004), 0.005)),
        "friction_estimate": float(np.clip(condition.friction + rng.normal(0, 0.045), 0.15, 1.0)),
        "position_sigma_m": float(condition.pos_mm / 1000 * factor),
        "orientation_sigma_rad": float(np.deg2rad(condition.rot_deg) * factor),
        "spacing_m": condition.spacing_mm / 1000,
    }


def latent_success(obs: dict, candidate: GraspCandidate, condition: Condition, attempt: int, rng: np.random.Generator) -> tuple[bool, str, dict]:
    label = obs["true_shape"] if obs["true_shape"] in CLASSES else obs["predicted_label"]
    ideal_width = {"circle": .048, "square": .051, "triangle": .043}[label]
    width_term = -abs(candidate.width_m - ideal_width) / .0065
    yaw = abs(((candidate.yaw_offset_rad + np.pi / 2) % np.pi) - np.pi / 2)
    if label == "triangle": yaw_term = -abs(yaw - np.pi / 2) / 1.1
    elif label == "square": yaw_term = -min(yaw, abs(yaw - np.pi / 2)) / 1.2
    else: yaw_term = -0.08 * yaw
    pos_error = abs(rng.normal(0, condition.pos_mm / 1000))
    correction = math.hypot(candidate.dx_m, candidate.dy_m)
    pos_term = -(max(pos_error - 0.55 * correction, 0) / .018) ** 1.25
    force_need = 1.2 + 0.8 * (0.55 / max(condition.friction, .18))
    force_term = -max(force_need - candidate.force_n, 0) * 1.05 - max(candidate.force_n - 2.75, 0) * .5
    crowd = max((65 - condition.spacing_mm) / 30, 0)
    depth_penalty = condition.depth_mm / 12
    unknown_penalty = 0.18 if not obs["known"] else 0.0
    logit = 2.55 + width_term + yaw_term + pos_term + force_term - 0.55 * crowd - 0.45 * depth_penalty - unknown_penalty + 0.30 * attempt
    p_contact = 1 / (1 + math.exp(-logit))
    contact = rng.random() < p_contact
    if not contact:
        failure = "empty_grasp" if pos_error > .009 or abs(candidate.width_m - ideal_width) > .008 else "contact_miss"
        return False, failure, {"p_contact": p_contact, "finger_width": 0.0015, "table_clear": False, "held": False, "placed": False}
    slip_prob = float(np.clip(0.30 * max(0.55 - condition.friction, 0) / .30 + 0.16 * max(force_need - candidate.force_n, 0) + 0.06 * crowd, 0.01, 0.72))
    held = rng.random() >= slip_prob
    finger_width = float(np.clip(ideal_width * rng.uniform(.20, .42), .0058, .022))
    table_clear = bool(rng.random() > 0.025)
    dual_ok = finger_width >= .0055 and table_clear
    if not dual_ok: return False, "verification_failed", {"p_contact": p_contact, "finger_width": finger_width, "table_clear": table_clear, "held": held, "placed": False}
    if not held: return False, "slip", {"p_contact": p_contact, "finger_width": finger_width, "table_clear": table_clear, "held": False, "placed": False}
    place_prob = float(np.clip(.985 - condition.pos_mm / 180 - crowd * .05, .65, .99))
    placed = rng.random() < place_prob
    return bool(placed), ("none" if placed else "placement_outside_bin"), {"p_contact": p_contact, "finger_width": finger_width, "table_clear": table_clear, "held": True, "placed": placed}


def train_models(out: Path, seed: int, rows: int) -> tuple[SuccessPredictor, GeometryNoveltyDetector, dict]:
    rng = np.random.default_rng(seed)
    observations=[]; candidates=[]; attempts=[]; labels=[]; geometry=[]; geometry_labels=[]
    # Geometry support is calibrated on independent known-shape observations.
    for _ in range(3600):
        shape = rng.choice(CLASSES); o = observe(shape, TRAIN_CONDITION, rng)
        geometry.append(o); geometry_labels.append(shape)
    novelty = GeometryNoveltyDetector(.985).fit(geometry, geometry_labels)
    for _ in range(rows):
        condition = Condition("train_randomized", float(rng.uniform(1, 8)), float(rng.uniform(0, 12)), float(rng.uniform(.5, 3.5)), float(rng.uniform(.45, .85)), float(rng.uniform(70, 130)), CLASSES, "train")
        shape = rng.choice(CLASSES); obs = observe(shape, condition, rng)
        nov = novelty.evaluate(obs, obs["predicted_label"], obs["confidence"]); obs.update(nov)
        cand = rng.choice(generate_candidates(obs, int(rng.integers(0, 3))))
        attempt = int(rng.integers(0, 3)); success, _, _ = latent_success(obs, cand, condition, attempt, rng)
        observations.append(obs); candidates.append(cand); attempts.append(attempt); labels.append(int(success))
    split = int(rows * .8)
    predictor = SuccessPredictor(seed).fit(observations[:split], candidates[:split], attempts[:split], labels[:split], split="train", held_out_rows=rows-split)
    y = np.asarray(labels[split:]); p = np.asarray([predictor.predict_success(o,[c],a)[0] for o,c,a in zip(observations[split:],candidates[split:],attempts[split:])])
    diagnostics = {"rows": rows, "train_rows": split, "test_rows": rows-split, "test_positive_rate": float(y.mean()), "roc_auc": float(roc_auc_score(y,p)), "average_precision": float(average_precision_score(y,p)), "brier": float(brier_score_loss(y,p))}
    predictor.metadata.update(diagnostics)
    predictor.save(out/"models"/"grasp_success_hgb.joblib"); novelty.save(out/"models"/"geometry_novelty.joblib")
    return predictor, novelty, diagnostics


def choose_baseline(obs: dict, candidates: list[GraspCandidate], attempt: int) -> GraspCandidate:
    # Same deterministic rule used by the prior advanced edition: centered grasp,
    # then yaw/offset/force recovery on later attempts.
    if attempt == 0: return candidates[0]
    return candidates[min(3 + attempt, len(candidates)-1)]


def run_object(run_seed: int, config_name: str, cfg: dict, condition: Condition, shape: str, predictor: SuccessPredictor, novelty: GeometryNoveltyDetector) -> dict:
    rng = np.random.default_rng(run_seed)
    obs = observe(shape, condition, rng, no_tta=cfg.get("no_tta", False))
    n = novelty.evaluate(obs, obs["predicted_label"], obs["confidence"])
    if cfg.get("geometry_novelty", False): obs.update(n)
    else:
        # Probability-only rejection: intentionally separate for ablation.
        obs.update({"unknown": obs["confidence"] < .55, "novelty_score": n["novelty_score"], "novelty_p_value": n["novelty_p_value"]})
    reobs = 0; time_s = 1.2; attempts = 0; first_success = False; accepted = False; final_success=False; failure="none"; selected="none"; predicted_p=0.0
    while attempts < 3:
        candidates = generate_candidates(obs, attempts)
        if cfg.get("learned", False): cand, ps = predictor.select(obs, candidates, attempts); predicted_p=float(np.max(ps))
        else: cand=choose_baseline(obs,candidates,attempts); predicted_p=.62
        decision = decide(obs, predicted_p, reobs, cfg.get("reobserve",False), cfg.get("unknown_hold",False))
        if decision.action == Action.REOBSERVE:
            reobs += 1; time_s += 2.8
            obs = observe(shape, condition, rng, reobserved=True, no_tta=cfg.get("no_tta",False))
            n = novelty.evaluate(obs, obs["predicted_label"], obs["confidence"])
            obs.update(n if cfg.get("geometry_novelty",False) else {"unknown": obs["confidence"] < .55, "novelty_score": n["novelty_score"], "novelty_p_value": n["novelty_p_value"]})
            continue
        if decision.action == Action.HOLD:
            failure="held_unknown" if not obs["known"] else "held_low_utility"; time_s += .5; break
        accepted=True; selected=cand.candidate_id; attempts += 1; time_s += 8.2 + .7*attempts
        success, failure, signals = latent_success(obs,cand,condition,attempts-1,rng)
        if attempts == 1: first_success=success
        if success:
            # Correct-bin success requires a known shape, correct classification, hold
            # through transport, and final pose inside the corresponding bin.
            final_success = bool(obs["known"] and obs["predicted_label"] == shape and signals["held"] and signals["placed"])
            if not final_success and obs["known"] and obs["predicted_label"] != shape: failure="misclassified_bin"
            break
        time_s += 2.0  # recovery/retreat
    held = not accepted
    mis_sort = bool(obs["known"] and accepted and obs["predicted_label"] != shape)
    unknown_false_accept = bool((not obs["known"]) and accepted)
    return {"seed":run_seed,"configuration":config_name,"condition":condition.name,"split":condition.split,"shape":shape,"known":obs["known"],"predicted_label":obs["predicted_label"],"confidence":obs["confidence"],"novelty_score":obs["novelty_score"],"unknown_decision":bool(obs["unknown"]),"accepted":accepted,"held":held,"correct_bin":final_success,"first_grasp_success":first_success,"attempts":attempts,"mis_sort":mis_sort,"unknown_false_accept":unknown_false_accept,"time_s":time_s,"reobservations":reobs,"failure":failure,"candidate_id":selected,"predicted_grasp_success":predicted_p}


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int=2000) -> tuple[float,float]:
    if len(values)==0: return (float("nan"),float("nan"))
    idx=rng.integers(0,len(values),size=(samples,len(values))); means=values[idx].mean(axis=1)
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))


def aggregate(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows=[]; rng=np.random.default_rng(seed+991)
    for (cfg,cond),g in df.groupby(["configuration","condition"],sort=True):
        known=g[g.known]; unknown=g[~g.known]
        metrics={"correct_bin_rate":known.correct_bin.mean(),"first_grasp_success_rate":known.first_grasp_success.mean(),"mean_attempts":known.attempts.mean(),"mis_sort_rate":known.mis_sort.mean(),"hold_rate":g.held.mean(),"known_hold_rate":known.held.mean(),"mean_time_s":g.time_s.mean(),"mean_reobservations":g.reobservations.mean(),"unknown_false_accept_rate":unknown.unknown_false_accept.mean() if len(unknown) else np.nan,"n":len(g),"n_known":len(known),"n_unknown":len(unknown)}
        for metric in ("correct_bin","first_grasp_success","mis_sort","held","time_s","reobservations"):
            vals=(known if metric in {"correct_bin","first_grasp_success","mis_sort"} else g)[metric].astype(float).to_numpy(); lo,hi=bootstrap_ci(vals,rng); metrics[f"{metric}_ci_low"]=lo; metrics[f"{metric}_ci_high"]=hi
        rows.append({"configuration":cfg,"condition":cond,**metrics})
    return pd.DataFrame(rows)


def plot_results(summary: pd.DataFrame, out: Path) -> None:
    order=[c.name for c in EVAL_CONDITIONS]
    configs=[c for c in ("A","B","C","D") if c in set(summary.configuration)]
    fig,axes=plt.subplots(2,2,figsize=(15,9)); metrics=[("correct_bin_rate","Correct-bin success"),("first_grasp_success_rate","First-grasp success"),("mean_time_s","Mean task time [s]"),("unknown_false_accept_rate","Unknown false acceptance")]
    for ax,(m,title) in zip(axes.ravel(),metrics):
        for cfg in configs:
            x=summary[summary.configuration==cfg].set_index("condition").reindex(order); ax.plot(range(len(order)),x[m],marker="o",label=cfg)
        ax.set_title(title); ax.set_xticks(range(len(order)),order,rotation=32,ha="right"); ax.grid(alpha=.25)
    axes[0,0].legend(title="Configuration",ncol=4); fig.tight_layout(); fig.savefig(out/"figures"/"abcd_comparison.png",dpi=180); plt.close(fig)


def make_video(records: pd.DataFrame, out: Path) -> None:
    """Create a compact, explicitly labelled benchmark replay video."""
    chosen=[]
    for cfg in ("A","B","C","D"):
        part=records[(records.configuration==cfg)&(records.condition.isin(["combined_unseen","open_set"]))].head(8); chosen.extend(part.to_dict("records"))
    w,h=1280,720; fps=24; writer=cv2.VideoWriter(str(out/"videos"/"adaptive_benchmark_replay.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    colors={"A":(70,110,220),"B":(75,185,90),"C":(220,165,45),"D":(190,80,180)}
    for rec in chosen:
        for f in range(48):
            img=np.full((h,w,3),245,np.uint8); cv2.rectangle(img,(0,0),(w,74),(30,36,48),-1)
            cv2.putText(img,"ADAPTIVE SORTING - CONTROLLED BENCHMARK REPLAY",(32,46),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2,cv2.LINE_AA)
            cfg=rec["configuration"]; cv2.putText(img,f"Configuration {cfg}  |  {rec['condition']}",(40,116),cv2.FONT_HERSHEY_SIMPLEX,.8,colors[cfg],2)
            cv2.putText(img,f"shape={rec['shape']}  predicted={rec['predicted_label']}  candidate={rec['candidate_id']}",(40,154),cv2.FONT_HERSHEY_SIMPLEX,.65,(45,45,45),2)
            # workspace, source object, bins
            cv2.rectangle(img,(80,210),(850,640),(180,180,180),2)
            bin_names=CLASSES
            for i,name in enumerate(bin_names):
                x=930+i*105; cv2.rectangle(img,(x,235),(x+85,320),colors.get(cfg,(80,80,80)),2); cv2.putText(img,name[0].upper(),(x+31,288),cv2.FONT_HERSHEY_SIMPLEX,1,(40,40,40),2)
            start=np.array([260,430]); target=np.array([970+105*CLASSES.index(rec['predicted_label']),275])
            t=min(max((f-8)/28,0),1); pos=(start*(1-t)+target*t).astype(int)
            cv2.circle(img,tuple(pos),30,(80,130,210),-1); cv2.circle(img,tuple(pos),30,(30,30,30),2)
            cv2.line(img,(250,120),(int(pos[0]),int(pos[1])),(55,55,55),10)
            action="HOLD" if rec["held"] else ("SORT OK" if rec["correct_bin"] else "RECOVER / FAIL")
            cv2.putText(img,f"decision: {action}",(40,685),cv2.FONT_HERSHEY_SIMPLEX,.8,(20,20,20),2)
            cv2.putText(img,"Visualization of logged benchmark outcomes; not Genesis footage",(690,690),cv2.FONT_HERSHEY_SIMPLEX,.48,(90,90,90),1)
            writer.write(img)
    writer.release()


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=ROOT/"out"/"experiments"); ap.add_argument("--seed",type=int,default=37); ap.add_argument("--objects-per-condition",type=int,default=200); ap.add_argument("--training-rows",type=int,default=12000); args=ap.parse_args()
    out=args.out; [p.mkdir(parents=True,exist_ok=True) for p in (out,out/"models",out/"tables",out/"figures",out/"logs",out/"videos")]
    t0=time.time(); predictor,novelty,diag=train_models(out,args.seed,args.training_rows)
    rows=[]
    for config_name,cfg in CONFIGS.items():
        for ci,condition in enumerate(EVAL_CONDITIONS):
            for i in range(args.objects_per_condition):
                shape=condition.shapes[i%len(condition.shapes)]
                common_seed=args.seed*10_000_000+ci*100_000+i
                rows.append(run_object(common_seed,config_name,cfg,condition,shape,predictor,novelty))
    records=pd.DataFrame(rows); summary=aggregate(records,args.seed)
    records.to_csv(out/"tables"/"object_level_results.csv",index=False); summary.to_csv(out/"tables"/"abcd_summary.csv",index=False)
    # Paired differences use exactly the same object seeds across configurations.
    pivot=records.pivot_table(index=["condition","seed"],columns="configuration",values=["correct_bin","first_grasp_success","time_s","reobservations"],aggfunc="first")
    deltas=[]
    for cond in [c.name for c in EVAL_CONDITIONS]:
        p=pivot.loc[cond]
        for cfg in ("B","C","D"):
            deltas.append({"condition":cond,"comparison":f"{cfg}-A","correct_bin_delta":float((p["correct_bin"][cfg].astype(float)-p["correct_bin"]["A"].astype(float)).mean()),"first_grasp_delta":float((p["first_grasp_success"][cfg].astype(float)-p["first_grasp_success"]["A"].astype(float)).mean()),"time_delta_s":float((p["time_s"][cfg]-p["time_s"]["A"]).mean()),"reobserve_delta":float((p["reobservations"][cfg]-p["reobservations"]["A"]).mean())})
    pd.DataFrame(deltas).to_csv(out/"tables"/"paired_differences.csv",index=False)
    # Ablation only on open-set and combined unseen to focus the causal tests.
    abrows=[]
    for ai,(name,cfg) in enumerate(ABLATIONS.items()):
        for ci,condition in enumerate([EVAL_CONDITIONS[6],EVAL_CONDITIONS[7]]):
            for i in range(args.objects_per_condition):
                shape=condition.shapes[i%len(condition.shapes)]; common_seed=args.seed*20_000_000+ci*100_000+i
                abrows.append(run_object(common_seed,name,cfg,condition,shape,predictor,novelty))
    ab=pd.DataFrame(abrows); absum=aggregate(ab,args.seed+1); ab.to_csv(out/"tables"/"ablation_object_results.csv",index=False); absum.to_csv(out/"tables"/"ablation_summary.csv",index=False)
    plot_results(summary,out); make_video(records,out)
    failures=records[~records.correct_bin].groupby(["configuration","condition","failure"]).size().reset_index(name="count"); failures.to_csv(out/"tables"/"failure_taxonomy.csv",index=False)
    environment={"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"opencv":cv2.__version__,"sklearn":__import__("sklearn").__version__,"seed":args.seed,"training_rows":args.training_rows,"objects_per_condition":args.objects_per_condition,"evaluation_conditions":[asdict(c) for c in EVAL_CONDITIONS],"training_domain":{"position_mm":"Uniform[1,8]","rotation_deg":"Uniform[0,12]","depth_noise_mm":"Uniform[0.5,3.5]","friction":"Uniform[0.45,0.85]","spacing_mm":"Uniform[70,130]","shapes":CLASSES},"benchmark_type":"controlled stochastic contact-outcome model; not Genesis physics","elapsed_s":time.time()-t0}
    (out/"environment.json").write_text(json.dumps(environment,indent=2,ensure_ascii=False),encoding="utf-8")
    manifest={"generated_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"model_diagnostics":diag,"novelty_threshold":novelty.threshold,"files":{}}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name!="manifest.json": manifest["files"][str(p.relative_to(out))]={"sha256":sha256(p),"bytes":p.stat().st_size}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"model":diag,"summary_rows":len(summary),"object_rows":len(records),"ablation_rows":len(ab),"out":str(out)},indent=2))

if __name__=="__main__": main()
