# 失敗経験とマルチ候補学習に基づく適応型AIロボット把持・仕分けシステム

Franka Panda + 固定上方 RGB-D カメラで `circle / square / triangle` を仕分ける既存上級編を、**不確実性と失敗経験から把持戦略を改善する適応型システム**へ拡張した実装です。

## デモ動画（提出用 YouTube）

- **実機・シミュレーション実行動画**: https://youtu.be/9Ba3pNvG2-Y

## 重要: 今回の検証境界

- **実装済み**: 多候補把持（XY / yaw / 指幅 / 力）、経験学習型成功予測、GRASP / REOBSERVE / HOLD 判断、幾何サポートによる未知検出、アーム退避後の再撮影、指幅+table-clear+運搬保持+配置後位置の検証、失敗別復帰、JSONL/CSV/JSON ログ。
- **実行済み**: 13 unit tests、合成 RGB-D E2E、固定 seed の A/B/C/D 6,400 object 比較、ablation 2,000 object、学習 12,000 candidate rows、64秒のログ再生動画。
- **実行済み（この環境）**: Genesis qualification `qualified=true`（torch 2.11.0+cu128 / genesis 1.4.5 / RTX 4090）。`genesis_sort_ultra.py --episodes 2` の物理実行とカメラ動画を取得。
- 実測: 詳細は `out/genesis/summary.json`。再現動画: `out/genesis/PAI_final_demo_repro_04_ultra.mp4`（12 episodes / 約89秒、success_rate 0.75）。
- パッケージ同梱の `AI_Shape_Sorting_Robot_Arm_Ultra_Benchmark_Replay.mp4` は **制御ベンチマークのログ再生**であり、Genesis映像ではないことに注意。
- `genesis_sort_ultra.py` には overhead 遮蔽・完了物体再選択ループ対策の実行時修正を反映済み。

## 構成

```text
adaptive_grasp.py                 # 8候補生成、学習型成功予測、経験JSONL
decision.py                       # grasp / reobserve / hold のリスク判断
novelty.py                        # class-conditional geometry support + conformal calibration
perception.py                     # RGB-D segmentation、3D逆投影、幾何特徴
shape_ai.py                       # MLP + morphology TTA
supervisor.py                     # 二重把持確認、failure-specific recovery
genesis3d/genesis_sort_ultra.py   # 既存IK/PDを維持した統合runner
genesis3d/qualify_genesis.py      # import -> rigid step -> Franka IK の実行ゲート
run_experiments.py                # A/B/C/D、ablation、未学習条件
make_report_assets.py             # 図表と失敗例の再生成
configs/ultra.json                # scene / physics / thresholds / evaluation
out/experiments/                  # model, object-level CSV, summary, figures, video, manifest
out/genesis/                      # qualification結果（本runは未通過）
report/                           # 詳細レポート
```

## 適応型制御

1. RGB-Dから物体マスク、3D位置、向き、幾何特徴を計算。
2. closed-set MLP確率とは別に、既知クラスの幾何分布から novelty score を計算。
3. 位置・向き・指幅・力の8候補を生成。
4. 観測特徴+候補+過去の成否で学習した `HistGradientBoostingClassifier` が各候補の成功確率を予測。
5. 不確実または成功見込みが低い場合、Frankaをhomeへ退避し固定上方カメラで再撮影。
6. 未知判定または全候補低効用なら保留。そうでなければ候補をIK/PDへ渡す。
7. 指幅 AND 元位置table-clearで把持を確認し、運搬中保持と配置後bin位置も確認。
8. 結果を `grasp_experience.jsonl` に残し、次回学習へ接続。

## 比較構成

| 構成 | 内容 |
|---|---|
| A | 既存実験03相当: ルール候補、再観測判断なし、未知保留なし |
| B | 学習型候補選択を追加 |
| C | B + arm-clear再観測判断 |
| D | C + 幾何ベース未知物体保留 |

## 再現

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python test_offline.py --out out/offline_ultra
python run_experiments.py --seed 37 --training-rows 12000 --objects-per-condition 200
python make_report_assets.py
```

### Genesis gate

```bash
pip install -r requirements-genesis.txt
python genesis3d/qualify_genesis.py --out out/genesis/qualification.json
# qualified=true を確認してからのみ本番実行
python genesis3d/genesis_sort_ultra.py --cpu --episodes 2
```

本番runnerは `configs/ultra.json` の `dt=0.01 s`, `substeps=4`, CPU target、seed 37を読み、既存のFranka IK/PDを維持します。モデルと結果の対応は `out/run_registry.json`、全ファイルhashは `out/experiments/manifest.json` を参照。

## 主要な実測結果（制御ベンチマーク）

- 既知形状全条件（各構成1,487 paired objects）: A 88.8% -> B 90.0% -> C 90.5% -> D 90.0% correct-bin。
- 初回把持: A 53.8% -> B 61.8% -> C 62.1% -> D 61.9%。
- open-set: A/B/C は未知を100%受け入れ、Dは5.3%まで低下。Dの既知形状correct-binは97.7%を維持。
- combined unseen: BはAより+5.0ptだが、C/Dの再観測は時間増または保守的holdを生み、万能な改善ではなかった。

これらは `run_experiments.py` の確率的接触モデル上の結果であり、Genesis物理成功率ではありません。詳細な区間推定、失敗分類、ablationはPDFとCSVにあります。

## 提出時の注意
- Genesis。
- 1-3分の実映像を撮り、YouTubeで限定公開: https://youtu.be/9Ba3pNvG2-Y
- 氏名・所属・メールアドレス等は成果物に含めない。
