import numpy as np
from adaptive_grasp import SuccessPredictor, generate_candidates
from decision import Action, decide
from novelty import GeometryNoveltyDetector


def obs(label="circle", shift=0.0):
    base={"circle":[.91,.99,.96,1.28],"square":[.785,.99,.97,.5],"triangle":[.60,.98,.83,.375]}[label]
    return {"predicted_label":label,"confidence":.9,"margin":.8,"entropy":.1,"novelty_score":0.1,"circularity":base[0]+shift,"solidity":base[1],"aspect":base[2],"vertices_norm":base[3],"height_m":.04,"depth_mad_m":.001,"edge_clearance_m":.05,"friction_estimate":.6,"position_sigma_m":.003,"orientation_sigma_rad":.1,"spacing_m":.09}


def test_candidates_cover_all_control_dimensions():
    c=generate_candidates(obs())
    assert len(c)==8
    assert len({x.yaw_offset_rad for x in c})>1
    assert len({x.width_m for x in c})>1
    assert len({x.force_n for x in c})>1
    assert any(abs(x.dx_m)+abs(x.dy_m)>0 for x in c)


def test_predictor_fits_and_selects():
    observations=[]; candidates=[]; attempts=[]; y=[]
    for i in range(80):
        o=obs("circle"); cs=generate_candidates(o); c=cs[i%len(cs)]
        observations.append(o); candidates.append(c); attempts.append(i%3); y.append(int(c.candidate_id in {"g0","g1","g2"}))
    model=SuccessPredictor(3).fit(observations,candidates,attempts,y)
    selected,p=model.select(obs(),generate_candidates(obs()))
    assert selected.candidate_id in {"g0","g1","g2"}
    assert p.shape==(8,)


def test_geometry_rejects_far_shape_even_at_high_confidence():
    rows=[]; labels=[]; rng=np.random.default_rng(4)
    for label in ("circle","square","triangle"):
        for _ in range(80):
            o=obs(label); o["circularity"]+=rng.normal(0,.006); rows.append(o); labels.append(label)
    detector=GeometryNoveltyDetector(.98).fit(rows,labels)
    unknown=obs("circle"); unknown.update(circularity=.35,solidity=.58,aspect=.55,vertices_norm=1.1,edge_damage=.3)
    result=detector.evaluate(unknown,"circle",.99)
    assert result["unknown"] is True


def test_decision_reobserve_and_hold():
    o=obs(); o["depth_mad_m"]=.009
    assert decide(o,.45,0,True,True).action==Action.REOBSERVE
    o["unknown"]=True
    assert decide(o,.9,0,True,True).action==Action.HOLD
