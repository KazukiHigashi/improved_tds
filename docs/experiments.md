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

```bash
python -m improved_tds.experiments.train_sb3 \
  --env-id ExpScissor1-v0 --algorithm sac \
  --total-timesteps 2000000 --seed 0 --device cuda \
  --output runs/scissors1/model \
  --dataset runs/scissors1/training_candidates.npz \
  --checkpoints-dir runs/scissors1/checkpoints
```

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
