# 1-3 minute Genesis video storyboard

| Time | Visual | Overlay |
|---|---|---|
| 0-10 s | Four-way A/B/C/D result teaser | Ultra Adaptive Edition / Genesis footage |
| 10-25 s | Overhead RGB-D detections | class probability + geometry novelty |
| 25-45 s | Eight grasp candidates | XY / yaw / width / force + P(success) |
| 45-65 s | Arm retreats and camera re-observes | REOBSERVE: fixed camera, arm clear |
| 65-90 s | Pick, lift, transport, place | finger width + table clear + hold + bin pose |
| 90-110 s | Unknown or damaged shape stays on table | HOLD: open-set geometry support |
| 110-135 s | Empty grasp or slip recovery | failure taxonomy -> changed candidate |
| 135-160 s | Comparison chart and limitations | same seeds; Genesis physics result only |

Rules:
- Keep total duration between 1 and 3 minutes.
- State 2x playback if used.
- Do not call `adaptive_benchmark_replay_h264.mp4` Genesis footage.
- Keep names, company, email, desktop notifications and access tokens out of frame.
