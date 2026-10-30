"""Record an honest Genesis environment qualification result.

Exit 0 only after importing Genesis and torch, creating a CPU scene, stepping a
plane + free box, loading Franka Panda, and solving a finite hand-link IK target.
"""
from __future__ import annotations
import argparse, json, platform, sys, traceback
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,required=True); a=p.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True)
    result={"qualified":False,"python":sys.version,"platform":platform.platform(),"checks":{}}
    try:
        import torch, genesis as gs
        result["torch_version"]=torch.__version__; result["genesis_version"]=getattr(gs,"__version__","unknown")
        result["checks"]["imports"]=True
        gs.init(backend=gs.cpu,seed=37)
        scene=gs.Scene(sim_options=gs.options.SimOptions(dt=.01,substeps=4),show_viewer=False)
        scene.add_entity(gs.morphs.Plane())
        box=scene.add_entity(gs.morphs.Box(size=(.04,.04,.04),pos=(.5,0,.04)))
        franka=scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        scene.build(); [scene.step() for _ in range(25)]
        pos=np.asarray(box.get_pos()); result["checks"]["rigid_body_finite"]=bool(np.isfinite(pos).all())
        hand=franka.get_link("hand"); q=np.asarray(franka.inverse_kinematics(link=hand,pos=np.array([.48,0,.28]),quat=np.array([0.,1.,0.,0.])))
        result["checks"]["franka_ik_finite"]=bool(np.isfinite(q).all())
        result["qualified"]=all(result["checks"].values())
    except Exception as exc:
        result["error_type"]=type(exc).__name__; result["error"]=str(exc); result["traceback_tail"]=traceback.format_exc().splitlines()[-8:]
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2)); raise SystemExit(0 if result["qualified"] else 2)

if __name__=="__main__": main()
