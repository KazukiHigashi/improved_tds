# 成功姿勢だけを使うTDS再評価

## 収集の変更

固定グリッド試行の終了をDB完成条件にしない。既存の同一Best Coverage Policyによる再収集記録から、保持条件を満たした成功姿勢だけを読み込む。失敗・保持不足・工具状態が目標範囲外の行は新DBに含めない。過去の実験ファイルは履歴として変更しない。

5つの固定Fit/Calibration/Test分割すべてについて、Fitの各角度区間に重複のない成功姿勢10件（全100件）が揃い、Calibration参照も揃うまで再試行する。足りない分割と角度区間を選び、そのFit初期姿勢だけから試行する。Test姿勢の成功でFitの不足を埋めない。分割・成功条件・重複閾値は変更しない。

同じ初期状態と同じ決定的入力を繰り返しても異なる姿勢は増えないため、区間内の目標値とreset seedを変え、同じ凍結Best Policyの決定的出力・確率的出力を交互に使用する。重みの更新や追加学習はしない。確率的試行も元の把持・誤差・低速度・保持条件に合格した場合のみ保存する。

`collect_until_tds_ready` の本実行はエピソード上限なし。25試行ごと、または成功姿勢追加時にチェックポイントと進捗JSONを保存する。再開時はpartial DBを使用できる。失敗試行は数値の監査ログに記録するだけで、失敗姿勢ベクトルをDBには保存しない。

Best Policyが不足区間に到達できない場合、回数を増やすだけで成功する保証はない。その場合も未充足を完成扱いにせず、無進捗試行数を表示して収集を継続する。必要数が揃ったという検査結果なしにTDS評価へ進まない。

候補DBの件数はタスクごとに異なるが、PCA/PLSが実際に使用するXは全15タスク・全5分割で100行、各区間10行に統一する。選択済みFit DBを別保存し、失敗姿勢や件数の水増しを含まないことを検査する。

## 把持軸と評価

Push 3で把持指標PLSが推定できず評価全体が停止した問題を修正。推定不能なforce_pls枝だけを理由付きで保存し、操作用PCA/PLSや残差PCAの枝は継続する。
今回のadaptive評価は**全Push/Sprayで残差PCA把持軸に統一**する。操作軸はPCAとPLSの比較、Scissorsは引き続き1自由度。前回のforce_pls把持軸とは異なる実験条件としてmanifestとHTMLに明示し、結果を混在させない。
成功姿勢の確保とFF制御のCalibration成立は別の検査であり、後者の不成立を隠したり基準を緩めたりしない。

## HTML

`comparison/index.html` にPCA/PLSのCoverage比較棒グラフ、タスクごとに左右の応答グラフ、成功Fit姿勢の散布図、Calibration/Test応答、左右のスイープ姿勢・20Step画像を掲載。左右の工具状態軸を共通化し、横軸の意味が同じ場合は横軸範囲も共通化する。
全4条件の詳細ページも維持する。N/A（未実施）とCalibration不成立のCoverage 0を区別する。

実行例：

```powershell
../fdms_reconfiguration/.venv/Scripts/python.exe scripts/run_tds_20260918.py --output analysis/tds_success_only_20260918 --success-source-root analysis/tds_mixed_v3_20260918 --adaptive-grasp-method residual_pca --workers 4
```

## 開始前確認

118テスト合格。Scissors 5の実データで追加収集を検証し、58試行で全5分割の100件・角度区間条件を充足した。候補DBは成功姿勢のみ1,366件。これはFF制御のCalibration成立を保証するものではない。
