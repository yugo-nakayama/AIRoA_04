# Genesis qualification and execution checklist

## Gate 0 - dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-genesis.txt
python genesis3d/qualify_genesis.py --out out/genesis/qualification.json
```

Do not report a Genesis result unless `qualified` is `true`. The packaged run is
`false` because the managed package feed did not provide torch.

## Gate 1 - frozen settings

- Genesis target: 0.3.3; torch target: 2.7.1 (targets, not validated in the packaged run)
- CPU backend, seed 37
- `dt=0.01 s`, `substeps=4`, gravity `[0,0,-9.81] m/s2`
- Camera 640x480, vertical FOV 50 deg, fixed overhead
- Franka MJCF `xml/franka_emika_panda/panda.xml`
- Record `pip freeze`, GPU/driver if used, and SHA-256 of config/models.

## Gate 2 - one-object dry run

```bash
python genesis3d/genesis_sort_ultra.py --cpu --vis --episodes 1
```

Confirm depth units, table-depth estimate, camera transform, finite IK, finger DOF
indices, object collision geometry, negative closing-force convention, and bin reachability.

## Gate 3 - verification signals

For one successful and one failed grasp, inspect JSONL for:

1. selected candidate + all predicted probabilities;
2. finger width after lift;
3. original location table-clear;
4. width after transport (slip test);
5. final object-to-bin distance;
6. appended labelled experience.

## Gate 4 - A/B/C/D physics run

Use identical scene seeds for all configurations. Run nominal, each single-factor
shift, combined unseen and open set. Keep training scenes disjoint. Never combine
surrogate and Genesis rows under the same `benchmark_type`.

## Gate 5 - submission video

Record a 1-3 minute Genesis camera video. Show candidate score, arm-clear
re-observation, a successful placement, an unknown hold and one recovery. Label any
speed-up. The packaged benchmark replay is not a substitute for physical footage.
