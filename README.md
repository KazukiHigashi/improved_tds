# Improved TDS — 論文用実験リポジトリ

この作業ツリーでは、Scissors・Push・Spray各5タスクについて、成功姿勢DBから作るPCA/PLSの1自由度TDS、FF/FB制御、および未知ツール評価の**旧・統一学習条件**を論文の主データとして採用する。高角度寄りに再学習した新Spray条件は主論文の結果に混ぜない。

## 論文成果物

- [論文用統合HTML](analysis/paper_ff_fb_generalization_20261008/index.html)：方法、実験条件、グラフ、表、画像、制約、および実験・スクリプト・結果ファイルの対応表。
- [選定済み15タスクbundle](analysis/paper15_paper_bundle_20260928/manifest.json)：採用したモデル・成功姿勢DB・TDSレポートのSHA-256を記録。
- [統一条件の学習](runs/paper15_unified_3m_slew_20260927/manifest.json)：15タスクの元学習結果。
- [共通9目標評価](analysis/matched_nine_goals_clipped_20261008/summary.json)：FF、未知ツール、選定4タスクのFBを共通目標で照合。
- [研究倫理・実装監査](docs/claude_external_audit_20261008.md)：主張、情報境界、較正と評価の分割。
- 実機用の試作コードは `scripts/real_experiment/` に保持する。実機安全性は未検証である。

論文HTMLは次で再生成できる（依存ライブラリ、MuJoCo環境、およびローカルに保持した元実験データが必要）。GitにはHTML・集計JSON・生成コードを記録するが、容量の大きい学習モデル、姿勢DB、画像、個別試行JSONは原則として含めない。このため、Gitのクローン単独では画像や元データへのリンクをすべて表示・再計算できない。論文提出用に図版・元データを配布する際は別途アーカイブが必要である。

```powershell
$env:PYTHONPATH = 'src;scripts'
python scripts/build_paper_ff_fb_generalization_report.py
python -m pytest -p no:cacheprovider tests/test_matched_nine_protocol.py tests/test_training_reward_report.py tests/test_initial_state_gallery.py -q
```

## ディレクトリの扱い

`analysis/` と `runs/` には、上記の論文用結果とその直接の生成元だけを残す。過去の探索、スモークテスト、高角度寄りの新Spray条件は `Obsoleted/` に元の階層を保って退避した。**削除ではない。** `Obsoleted/` は `.gitignore` でGit管理外のため、別マシンへ移す場合は別途バックアップする。退避一覧は `Obsoleted/move_manifest.json` に記録する。

共有ライブラリ `src/`、MuJoCo資産 `configs/`、実機用 `scripts/real_experiment/` は依存を壊さないため現位置に残す。`scripts/` は論文実験の生成・再評価・集計に関係するコードを残し、旧キャンペーンや診断用コードを `Obsoleted/scripts/` に退避した。実験ごとの残存コードと出力ファイルの対応は統合HTMLの「実験・スクリプト・結果ファイルの対応」節に記載する。

`analysis/pytest_tds_update_20260925*` の4ディレクトリとルートの `.pytest_cache` は、現在のユーザー権限で内容の一覧も移動も拒否されるため未整理である。権限を勝手に変更・削除していない。
