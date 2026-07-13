# 軌道データセット

`TrajectoryStep`は、同期されたタイムスタンプ、step番号、関節位置・速度、action、ツール状態・目標状態・状態速度、任意のトルク・電流・接触データ、phase、報酬、瞬間成功、連続成功、エピソード終了フラグを保存する。`TrajectoryMetadata`は、機構族、インスタンス、タスク、成否、seed、episode ID、シミュレータパラメータ、コントローラ、checkpoint、DBの役割を保存する。新しい`.npz`ファイルには、`schema_version=1.1`、エピソードのoffset、JSONメタデータ、密配列を格納し、pickle化したPythonオブジェクトは含めない。保存は一時ファイルからのatomic renameで確定する。

検証処理では、空データ、NaN・無限大、単調でないタイムスタンプ、不正なepisode offset、一貫しない特徴量次元、フィールド長の不一致を拒否する。`SuccessfulTrajectoryCollector`は既定で、指定回数連続して成功したepisodeだけを候補DBへcommitする。デバッグ目的では失敗episodeも明示的に保存できるが、`samples()`が返すのは成功軌道だけである。`success_steps_only=True`では瞬間成功stepを、`stable_success_samples()`では各episodeの最初の安定成功窓だけを抽出する。

学習中に保存する`training_candidate`と、固定checkpointから目標binごとに成功数を揃えて収集する`formal_balanced_fit`、`formal_balanced_calibration`、`formal_balanced_test`は区別する。PCAのfit、tool状態校正、TDS評価には対応する分割だけを使用し、学習中の探索dataや別分割を混入させない。正式DBは全binで成功quotaを満たしたことをsummaryの`quota_met=true`で確認してから使用する。

schema 1.0は後方互換で読み込めるが、目標状態が存在しないため`goal_conditioned_samples()`は明示的に失敗する。欠損目標をゼロとして補完してはならない。

レガシーファイルは`[postures, tool_angles]`形式のobject配列である。次のコマンドで明示的に変換する。

```bash
python -m improved_tds.experiments.convert_legacy \
  synergy_dataset.npy trajectories.npz --instance-id scissors-1
```

レガシーの終端サンプルには速度や接触の履歴がない。変換器は測定値を推測して補完せず、この制約をメタデータに記録したうえで、1ステップの終端軌道として変換する。
