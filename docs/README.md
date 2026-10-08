# docs 索引

最終更新: 2026-10-06

論文の主実験に採用するPCA／PLSの1自由度TDS条件と数値は
[paper_one_dof_baseline_20261006.md](paper_one_dof_baseline_20261006.md) を参照してください。
15タスク実験基盤の実装・採用結果・従来のGit境界は
[project_snapshot_20260929.md](project_snapshot_20260929.md) に保存しています。
第2軸フィードバックの新規実験条件は
[second_axis_feedback_experiment_20261006.md](second_axis_feedback_experiment_20261006.md) にまとめています。

## いま動いている検討

- [synergy_analysis_plan.md](synergy_analysis_plan.md) —
  学習後にシナジーを取り出す手順と、その実施結果。仮説、データの取り方、
  チェックポイントの選び方、PCAとPLSの比較まで。
- [related_work_pls_synergy.md](related_work_pls_synergy.md) —
  PLSによるシナジー抽出の先行研究調査と、論文執筆への反映案。
  引用漏れと、査読で問われる論点への備え。
- [pls_tool_control_design.md](pls_tool_control_design.md) —
  取り出したシナジー1本でツールを動かす制御の設計と評価。
  TDS論文のはさみ手順を、握りが要るツールへ広げたもの。

## 仕様・設計

- [tds_math.md](tds_math.md) — TDSの数式と前提。
- [tds_research_workflow.md](tds_research_workflow.md) —
  研究報告としての実験設計、統計、図表、実行コマンド。
- [controllers.md](controllers.md) — コントローラと安全機構の責務。
- [datasets.md](datasets.md) — 軌道データセットの形式。
- [real_robot_integration.md](real_robot_integration.md) — 実機統合の境界。
- [implementation_plan_closed_loop_tds.md](implementation_plan_closed_loop_tds.md) —
  閉ループTDSの実装計画とリポジトリ調査（2026-07）。
- [experiments.md](experiments.md) —
  `tds_research_workflow.md` への案内のみ。README からの参照があるため残している。

## 手順書

- [initial_pose_authoring.md](initial_pose_authoring.md) —
  初期姿勢の作成とランダム初期姿勢DBの構築。
- [spray_lite_training_setup.md](spray_lite_training_setup.md) —
  Spray Lite の SAC+HER 学習設定。
- [scissors_tds_validation.md](scissors_tds_validation.md) —
  はさみのTDS制御の検証手順。

## 調査記録

日付入りの記録は [reports/](reports/) にまとめた。当時の状況を残すもので、
現在の設計は上の「いま動いている検討」を見る。

- [reports/push1_training_failure_report_20260811.md](reports/push1_training_failure_report_20260811.md)
- [reports/spray_training_failure_report_20260814.md](reports/spray_training_failure_report_20260814.md)
- [reports/spray_drivability_analysis_20260815.md](reports/spray_drivability_analysis_20260815.md)

## その他

- [handoff_context.md](handoff_context.md) —
  別のエージェントへ作業を渡すときの文脈共有用プロンプト（2026-08-14時点）。
  内容は当時のもので、その後の変更は反映していない。
