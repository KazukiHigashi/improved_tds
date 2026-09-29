# Scissors の振動抑制用 `smooth` モード

Viewer で旧 `simple` Best Policy の行動が急反転し、接触による把持判定に刃も含まれていることが判明した。この変更は旧実験の再現性を維持するため、別の制御バージョンとして追加した。旧モデルと新モデルの成功率は同一タスク定義の結果として直接比較しない。

旧Best Policyの短い診断（Scissors 3～5、各18試行）では、隣接ステップの正規化行動変化が0.5を超える成分の割合はそれぞれ約24%、21%、4%で、把持フラグは各タスクとも約3割のステップで反転した。Scissors 5は大半のステップで指令が飽和・停滞し、散発的に大きく跳ぶ。これは振動の定量的な兆候だが、実機安全性を直接測った値ではない。またScissors 4の旧Viewerの6試行では、6件が旧基準で成功し、そのうち1件の終了時にはミラーハンドルへの指接触がゼロであった。旧判定に刃接触が混入する影響を示す例である。

## Push・Spray との対応

Push と Spray の生環境には `relative_action=True` 時にアクチュエータ可動範囲の半幅の `action_delta_fraction=0.05` ずつ指令を更新する機構がある。現行の学習経路では両者ともデフォルトの `relative_action=False` を使用しているため、この機構は実際には有効ではない。三カテゴリに共通の `ControlRefinement` 観測と行動変化ペナルティは有効である。

Scissors `smooth` は絶対目標値の意味を保ちつつ、実際に適用する正規化目標値の変化を各制御ステップで `0.05` 以下にする。`0.05` は Push・Spray の任意相対モードの既定値に合わせたもの。要求値と適用値の二乗差を報酬で減点し、適用値を次の観測へ入れる。この制限は各指のアクチュエータ目標値に対するもので、関節速度・力・接触衝撃の安全上限ではない。

把持判定は少なくとも二本の異なる指が両方のハンドルに接触することを要求する。刃だけの接触は成功に数えない。Scissors 1・2 の短いメッシュ列と3～5の長いメッシュ列に対応する。

## 実行例

既存モデルにこのモードを後付けするのではなく、新規に学習する。

```powershell
python scripts/run_scissors_retention_v4.py --version smooth --output runs/scissors_smooth_example --variants '3 4 5' --timesteps 3000000 --n-envs 4 --collect-successes
```

学習後のViewer確認:

```powershell
python scripts/view_reach_collection.py --control-version smooth --run-root runs/scissors_smooth_example --variant 4 --episodes 20
```

Viewerの出力には各ステップの適用指令最大変化 `applied_delta` と制限された成分比 `limited` を表示する。実機利用には別途、関節速度・トルク・接触力の監視と独立した非常停止が必要。
