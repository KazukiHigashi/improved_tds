# 論文用リポジトリ整理（2026-10-11）

作業ブランチ: `chore/paper-only-cleanup-20261011`。既存の未コミット変更を保持したまま作業した。論文主データは旧・統一学習条件であり、高角度寄りの新Spray条件は採用しない。

## 残した入力と成果物

- 主報告: `analysis/paper_ff_fb_generalization_20261008/`
- 15タスク選定bundle: `analysis/paper15_paper_bundle_20260928/`
- 元学習: `runs/paper15_unified_3m_slew_20260927/`
- 共通9目標: `analysis/matched_nine_goals_clipped_20261008/`
- FB: `analysis/second_axis_feedback_effect_20261006/`
- 未知ツール関連: `analysis/initial_pca2_nn_20261008/`、`analysis/few_shot_axis_adaptation_20261008/`、`analysis/few_shot_guarded_20261008/`
- 採用レポートの生成元と図: `analysis/paper15_no_contact_floor_on_20260928/`、`analysis/paper15_unified_tds_corrected_20260928/`、`analysis/scissors_tds_palm_radius_1p5_20260928/`、`analysis/tds_viability_admitted_20260929/`

上記以外の分析・学習ディレクトリ、旧資料、旧実験テストと一部の一時ファイルは、元階層を保って `Obsoleted/` に移した。新Spray学習・TDS・未知ツール結果もそこにある。さらに `scripts/` を個別確認し、旧キャンペーンや診断用の52項目を `Obsoleted/scripts/` に退避した。論文用の生成・再評価・集計スクリプトとその直接依存、将来実機用の `scripts/real_experiment/` は残した。削除は行っていない。`Obsoleted/` はGit追跡対象外であり、別マシンでは再現されない。現地の退避一覧は `Obsoleted/move_manifest.json` を参照する。

主HTMLの「実験・スクリプト・結果ファイルの対応」節には、各実験の実施内容、実行・集計コード、出力ディレクトリ、代表JSONへのリンク、結果を掲載した節の対応を載せた。生成コードは存在しない参照先を検出して停止する。

## 未完了と理由

- `src/` は実験間の共有ランタイムとして現位置に保持した。旧補助モジュールの独立した退避は今回実施していない。
- `analysis/pytest_tds_update_20260925*` 4件と `.pytest_cache` はOSのアクセス拒否で内容の一覧・移動ができず、現位置に残った。権限変更や強制削除は行っていない。退避先に同名の空ディレクトリが2件作られているが、実データは移っていない。
- 既存の未コミット変更を含むため、このブランチはコミット・マージしていない。

## 検証

論文HTMLを再生成し、新Spray実験への節・リンクがないことと、実験対応表のファイル参照がすべて実在することを確認した。HTML内の相対ファイルリンク・画像リンクの実在確認を含むテストを実施した。主なテストは `test_matched_nine_protocol.py`、`test_training_reward_report.py`、`test_initial_state_gallery.py`、`test_paper_report_integrity.py` である。
