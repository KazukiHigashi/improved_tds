# 実験ワークフロー

研究報告としての完全な実験設計、数式、比較法、統計、図表、実行コードは
[TDS-based tool control研究報告の実験ワークフロー](tds_research_workflow.md)を参照する。

シミュレーション依存と開発依存は、`python -m pip install -e '.[dev]'`でインストールする。収集・学習コマンドではseedを明示的に指定する。

```bash
python -m improved_tds.experiments.collect \
  --env-id TDS-Trigger-v0 --episodes 20 --seed 0 \
  --output runs/trigger/data.npz

python -m improved_tds.experiments.fit_tds \
  --dataset runs/trigger/data.npz --method pca \
  --output runs/trigger/tds_pca.npz

python -m improved_tds.experiments.fit_tds \
  --dataset runs/trigger/data.npz --method pls \
  --output runs/trigger/tds_pls.npz

python -m improved_tds.experiments.calibrate \
  --dataset runs/trigger/data.npz --model runs/trigger/tds_pls.npz \
  --method isotonic --shots 6 --output runs/trigger/calibration.npz

python -m improved_tds.experiments.evaluate \
  --dataset runs/trigger/data.npz --model runs/trigger/tds_pls.npz \
  --calibration runs/trigger/calibration.npz

python -m improved_tds.experiments.evaluate_transfer \
  --manifest configs/scissors/instances.json --method pls --shots 3,6,12 \
  --output runs/scissors/transfer.json
```

## ハサミ方策学習と姿勢DB

学習CLIはSAC+HERを既定とし、TD3+HER・DDPG+HERも比較用に選択できる。SACは方策のエントロピーを自動調整し、外部action noiseを使用しない。連続5 steps成功したepisodeを候補DBへ保存する。stable successはHERの終端には使用せず、履歴依存terminalとgoal relabelの不整合を避ける。

学習中は10万stepごとに全初期姿勢×目標比率0.1/0.3/0.5の固定Validationを行う。成功率95%以上を2回連続で達成するとEarly Stopする。10万stepごとの通常checkpointは`checkpoints/`へ、Validation成功率が過去最高（同率時は追従誤差が最小）のモデルは`best_policy.zip`へ保存し、全評価履歴を`validation_history.json`へ記録する。論文用test splitはこの停止判定に使用しない。

```bash
python -m improved_tds.experiments.train_sb3 \
  --env-id ScissorsVariant1-v0 --algorithm sac \
  --total-timesteps 2000000 --seed 0 --device cuda \
  --output runs/scissors1/model \
  --dataset runs/scissors1/training_candidates.npz \
  --checkpoints-dir runs/scissors1/checkpoints
```

`train_sb3` は `tool-pose-pipeline/grasps` の task/hand/variant 対応 pool を必須とし、
各 episode の reset ごとに初期把持姿勢をランダムに選ぶ。基準初期姿勢を方策に探索
させる経路や、poolを使わないfallbackは設けない。

push 1–5のLite学習は次で一括実行する。

```powershell
.\scripts\train_push_lite_tasks.ps1 -TotalTimesteps 1000000
```

各push episodeでは、全poseで駆動確認済みのプランジャストローク5–50%から目標深さを
一様に選ぶ。成功には
目標誤差が許容内であること、2指以上がボトル胴を把持していること、ボトルが落下・
転倒していないことを同時に要求し、5 step連続で満たした状態を安定成功として記録する。
pushのTool DoFはslide jointなので、ここでいう目標は角度ではなく押込み深さ[m]である。

pushの学習報酬はHERで再計算可能なdense rewardとする。非落下・非転倒stepでは、
目標深さへの近さ、目標に対する押込み進捗、保持指の接触数、親指とプランジャの接触を
それぞれ連続的に加点し、目標到達と必要把持接触を同時に満たすと成功bonusを加える。
既定値は次式である。

```text
r_track  = 2.0  * (1 - clip(|depth - target| / push_limit, 0, 1))
r_push   = 1.0  * clip(depth / target, 0, 1)
r_grasp  = 2.0  * clip(grasp_contact_count / 2, 0, 1)
r_thumb  = 0.25 * thumb_on_plunger
r_bonus  = 5.0  * instantaneous_success
r_total  = r_track + r_push + r_grasp + r_thumb + r_bonus
```

落下または転倒時は上記を加算せず、報酬を`-250`で上書きしてepisodeを終了する。
非失敗stepの報酬は0以上であるため、負報酬を避ける目的で早期転倒する利点はない。
各係数は`train_sb3`の`--push-*-reward-weight`、`--push-success-bonus`、
`--push-failure-penalty`で変更でき、学習summaryにも保存される。

学習後は固定checkpointから目標範囲をbin分割し、各binで同数の安定成功軌道が得られるまで正式DBを収集する。試行上限に達したbinがあればsummaryの`quota_met`が`false`になるため、そのDBを正式解析に使用しない。`fit`、`calibration`、`test`は別ファイル・別seed領域で収集する。

```bash
python -m improved_tds.experiments.collect_policy \
  --env-id ExpScissor1-v0 --algorithm sac \
  --model runs/scissors1/model.zip \
  --output runs/scissors1/formal_fit.npz --split fit \
  --target-min 0.05 --target-max 0.75 \
  --target-bins 8 --successes-per-target 20 \
  --max-attempts-per-target 200
```

正式DBからTDSをfitする場合、各episodeで最初に成立した連続成功窓だけを使用する。成功後に偶然滞在した長さの違いでepisodeの重みが変わることを防ぐ。

```bash
python -m improved_tds.experiments.fit_tds \
  --dataset runs/scissors1/formal_fit.npz \
  --stable-success-only --method pca \
  --output runs/scissors1/tds_pca.npz
```

学習summaryにはcandidate成功軌道数とaction飽和率を保存する。checkpoint選択、正式DB収集、TDS fit、評価のdataをepisode・seed単位で分離する。短時間の単体・シミュレーションテストは`pytest`で実行し、長時間RL学習はテストスイートへ含めない。
