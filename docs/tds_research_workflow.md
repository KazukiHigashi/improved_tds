<!-- math: mathjax -->

# TDS-based tool control研究報告の実験ワークフロー

## 0. この文書の位置付けと結論の範囲

本書は、5種類のハサミ環境でTool DoF Synergy（TDS）を構築・検証し、研究報告の図表と議論へ到達するための事前固定ワークフローである。対象は`ExpScissor1-v0`から`ExpScissor5-v0`であり、以下の三段階を区別する。

Gymnasium環境は本研究の固定ベンチマークであり、本ワークフローの途中で環境のreward、観測、action、asset、物理パラメータを拡充・変更しない。環境側の機能追加は専用リポジトリでversion管理し、本リポジトリには固定versionだけを取り込む。

1. **表現の妥当性**: 成功近傍の手姿勢から得た1次元TDS座標が、独立dataのTool DoFと対応するか。
2. **制御の妥当性**: TDS座標からTool DoF目標へ逆写像した姿勢指令で、目標角を追従できるか。
3. **フィードバックの寄与**: 同じTDS・同じ較正・同じepisode条件で、tool-state feedbackがfeedforwardのみより追従誤差を低減するか。

本実験が直接支持できる主張は「**本シミュレータの、成功近傍かつ評価した角度範囲において、1次元TDSを用いたTool DoF追従が成立するか**」である。切断品質、実機性能、未評価の外乱への一般的頑健性、あるいはRL seed全体への一般化は、この実験だけからは主張しない。

> 重要: `evaluate_scissors_tds`は現在、固定方策を再rolloutしてepisode内の代表姿勢からPCA・較正を行う**統合smoke評価**である。以下で定義する論文の主解析は、`collect_policy`で作る`fit`・`calibration`・`test`の正式DBを用いる。両者は同じ結果として混在させない。

## 1. 実験条件（実行前に固定する表）

結果を得てから条件を変更すると比較の意味が失われるため、表中の値・seed・commit・環境XMLの識別子をrun manifestへ保存する。`K_p`掃引などを追加する場合は、主解析と別の感度解析として扱う。

|区分|事前固定する条件|本runの標準値|目的|
|---|---|---:|---|
|対象task|`ExpScissor1-v0`〜`ExpScissor5-v0`|5 task|把持形態・作動指の異なるハサミで評価する|
|RL方策|SAC + HER|SAC, `copy_info_dict=True`, `ent_coef=auto`|把持判定をHER再ラベル報酬にも保持する|
|学習量|epoch, episode/epoch, step/episode|100, 100, 100|各task (10^6) environment steps|
|RL seed|taskごとのseed|1001〜1005|今回の主run。一般化には追加seedが必要|
|成功判定|(e_c<\epsilon_c) かつ把持内|\(\epsilon_c=0.01\ \mathrm{rad}\)|TDSに不適切な非把持姿勢を除外する|
|安定成功長|連続成功step数 (K)|5|一過性の交差を姿勢DBに入れない|
|Tool DoF bin|範囲・bin数|0.05–0.75 rad, 8 bins|特定の目標角への偏りを抑える|
|正式DB quota|binごとの成功軌道数|20|fit/calibration/testの各binを均等化する|
|TDS|推定器・次元|PCA, 1次元|主手法。比較にPLS/covariance/randomを使う|
|較正|\(c=f(\rho)\)|isotonic|単調な逆写像を安全範囲内で使う|
|制御器|FF / FB|\(K_p=0.5,K_i=K_d=0\)|feedback寄与の主比較|
|外乱|nominal / resistance|damping=0.2, torque=0.03|事前固定した単一点の抵抗増加条件|
|統計|bootstrap / permutation|各2,000回|episodeを統計単位にした95% CI|

### 1.1 比較法と比較の意味

主比較は次の三つである。

|比較|同一に保つもの|変えるもの|答える問い|
|---|---|---|---|
|PCA-TDS vs supervised TDS|正式DB split・1次元・較正法|方向の推定法（PCA / PLS / covariance）|非教師あり主成分でTool DoFに十分対応できるか|
|PCA-TDS vs random direction|平均姿勢・次元・較正・test split|方向 \(s\)|任意の1次元圧縮ではなくTDS方向に情報があるか|
|Feedforward vs feedback|\(\bar q,s,f\)、target列、reset seed、外乱|Tool DoF feedbackの有無|状態feedbackが追従誤差を低減するか|

SAC+HER、TD3+HER、DDPG+HERの比較は、TDSそのものではなく**姿勢DBを供給する方策学習器の感度解析**である。主結論をTDSに置く場合、同一成功基準・同一DB構成で比較し、RLの優劣とTDSの優劣を混同しない。

## 2. タスク、観測、成功姿勢DB

各taskはShadow Hand Liteがハサミを把持し、ハサミhinge角を目標へ動かすgoal-conditioned環境である。Tool DoFを

$$
c_t = q^{\mathrm{hinge}}_t \in \mathbb{R}
$$

とし、手の作動関節姿勢を\(q_t\in\mathbb{R}^{d}\)とする。ここで\(d\)は当該環境のactuator数（現行ハサミでは14）である。報酬の成功指標は、目標\(c_d\)に対して

$$
I_t = \mathbb{1}\left[|c_t-c_d|<\epsilon_c\right]
      \mathbb{1}\left[\text{in-grasp-space}_t\right]
$$

で与える。安定成功は、時刻\(t\)までの連続成功長\(r_t\)が

$$
r_t=\begin{cases}
r_{t-1}+1 & (I_t=1),\\
0 & (I_t=0)
\end{cases},\qquad
I_t^{\mathrm{stable}}=\mathbb{1}[r_t\ge K]
$$

を満たすことと定義する。採用する姿勢は成功episodeの全stepではない。最初に\(I_t^{\mathrm{stable}}=1\)となる時刻\(t^\star\)に対し、

$$
\mathcal W=\{t^\star-K+1,\ldots,t^\star\}
$$

だけをTDS sampleとする。この規則により、成功後に長く滞在したepisodeがPCAを過剰に支配することを防ぐ。\(\mathcal W\)が作れないepisodeは成功DBから除外し、除外数・成功率・binごとのquota達成状況を必ず報告する。

各taskは把持に寄与する指や初期形態が異なるため、TDSはtaskごとに独立にfitする。5 taskの姿勢を無条件にpoolして単一TDSをfitしてはならない。これは手の形態差をTool DoF相関と誤認する交絡を避けるためである。

## 3. 学習から正式DBまで

### 3.1 方策学習

今回の実行中batchは、各taskにつき100 epoch、すなわち\(100\times100\times100=10^6\) stepsである。実行コードは次である。

```bash
cd /path/to/improved-tds
systemctl --user status improved-tds-scissors-100e-20260713.service --no-pager
tail -n 40 runs/scissors_sac_100epochs_20260713/supervisor.log
```

個別の再実行は次のコードで行う。既存runを上書きせず、別の出力directoryを使用する。

```bash
.venv/bin/python -m improved_tds.experiments.train_sb3 \
  --env-id ExpScissor1-v0 --algorithm sac \
  --total-timesteps 1000000 --seed 1001 --device cpu \
  --output runs/scissors_sac_100epochs_20260713/exp_scissor1/model \
  --dataset runs/scissors_sac_100epochs_20260713/exp_scissor1/training_candidates.npz \
  --checkpoints-dir runs/scissors_sac_100epochs_20260713/exp_scissor1/checkpoints \
  --checkpoint-freq 100000 --dataset-save-frequency 100
```

学習候補DB（`training_candidate`）は探索中方策の副産物であり、正式なPCA・校正・testに使わない。checkpoint選択には、`model_training_summary.json`、action飽和率、候補成功数、学習曲線を確認する。成功率0のcheckpoint、NaN/Inf、異常に高いaction飽和はTDS評価へ進めない停止条件である。

### 3.2 fit / calibration / testの正式DB

正式DBは固定済みの`model.zip`を決定論的にrolloutして作る。各bin内のTool DoF目標は連続一様分布からsampleされ、quotaに達するまで収集する。同じ目標点を繰り返すのではない。各splitは別path・別seed領域・別`dataset_role`で保存する。

```bash
ROOT=runs/scissors_sac_100epochs_20260713
TASK=1
MODEL="$ROOT/exp_scissor${TASK}/model.zip"

for SPLIT in fit calibration test; do
  .venv/bin/python -m improved_tds.experiments.collect_policy \
    --env-id "ExpScissor${TASK}-v0" --algorithm sac --model "$MODEL" \
    --output "$ROOT/exp_scissor${TASK}/formal_${SPLIT}.npz" \
    --split "$SPLIT" --target-min 0.05 --target-max 0.75 \
    --target-bins 8 --successes-per-target 20 --max-attempts-per-target 200 \
    --seed 20000
done
```

各`formal_*.summary.json`の`quota_met`が`true`であることを確認するコードは以下である。

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("runs/scissors_sac_100epochs_20260713/exp_scissor1")
for split in ("fit", "calibration", "test"):
    summary = json.loads((root / f"formal_{split}.summary.json").read_text())
    assert summary["quota_met"], (split, summary["target_successes"])
    print(split, summary["successful_trajectories"], summary["target_successes"])
PY
```

`quota_met=false`、DB role不一致、またはbinごとの成功不足がある場合、そのtaskを「TDS不成立」と記録し、失敗episodeを追加して見かけのsample数を増やさない。

## 4. TDSのfitと比較

fit splitの\(N\)姿勢を行列\(Q\in\mathbb{R}^{N\times d}\)として、平均姿勢を

$$
\bar q = \frac{1}{N}\sum_{i=1}^{N}q_i
$$

とする。PCA-TDSは中心化行列\(X=Q-\mathbf{1}\bar q^\top\)のSVD

$$
X=U\Sigma V^\top
$$

から第1右特異ベクトル\(s=V_{:,1}\)を取る。TDS座標は

$$
\rho_i=s^\top(q_i-\bar q),\qquad
\hat q_i=\bar q+s\rho_i
$$

であり、説明分散比

$$
\mathrm{EVR}_1=\frac{\sigma_1^2}{\sum_j\sigma_j^2}
$$

と再構成RMSEを報告する。符号はfit splitだけで\(\mathrm{corr}(\rho,c)\ge0\)となるよう固定し、calibration/testを見て反転してはならない。

PCA-TDS、PLS-TDS、covariance-TDS、random directionを同一split・同一次元で比較する。PLS/covarianceはTool DoFを教師として方向を推定するため、PCA-TDSより高い相関でも「非教師ありTDSの優位」を示すものではない。random directionは、任意の1次元投影では説明できないことを確認するnegative controlである。

実行コードは以下である。`--stable-success-only`はsplit roleを検査するため、`formal_fit`以外を誤ってfitへ使うと停止する。

```bash
ROOT=runs/scissors_sac_100epochs_20260713/exp_scissor1

.venv/bin/python -m improved_tds.experiments.fit_tds \
  --dataset "$ROOT/formal_fit.npz" --stable-success-only \
  --method pca --output "$ROOT/tds_pca.npz"

.venv/bin/python -m improved_tds.experiments.fit_tds \
  --dataset "$ROOT/formal_fit.npz" --stable-success-only \
  --method pls --output "$ROOT/tds_pls.npz"

.venv/bin/python -m improved_tds.experiments.fit_tds \
  --dataset "$ROOT/formal_fit.npz" --stable-success-only \
  --method covariance --output "$ROOT/tds_covariance.npz"

.venv/bin/python -m improved_tds.experiments.fit_tds \
  --dataset "$ROOT/formal_fit.npz" --stable-success-only \
  --method random --seed 30001 --output "$ROOT/tds_random.npz"
```

## 5. 校正と独立test

TDS座標はTool DoFそのものではないため、calibration splitで単調関数

$$
c=f(\rho)
$$

をfitする。isotonic calibrationは単調性を保つが、平坦部分は逆関数が一意でない。そのため、逆写像\(f^{-1}\)はcalibratorが報告する可逆・安全範囲内だけで用い、外挿をしない。

```bash
ROOT=runs/scissors_sac_100epochs_20260713/exp_scissor1

.venv/bin/python -m improved_tds.experiments.calibrate \
  --dataset "$ROOT/formal_calibration.npz" --stable-success-only \
  --model "$ROOT/tds_pca.npz" --method isotonic --shots 0 \
  --output "$ROOT/calibration_isotonic.npz"

.venv/bin/python -m improved_tds.experiments.evaluate \
  --dataset "$ROOT/formal_test.npz" --stable-success-only \
  --model "$ROOT/tds_pca.npz" \
  --calibration "$ROOT/calibration_isotonic.npz"
```

このoffline評価は\(f(\rho)\)の予測誤差であり、閉ループ制御性能ではない。校正testでの指標は

$$
\mathrm{RMSE}_c=
\sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(c_i-f(\rho_i)\right)^2}
$$

である。PCAの相関・EVRが高くても、\(f^{-1}\)が可逆でなければ姿勢指令として使えない。このため相関、EVR、校正RMSE、安全範囲を別々に報告する。

## 6. TDS-based tool controlの比較

### 6.1 指令式

feedforward基準は

$$
\rho_{\mathrm{FF}}=f^{-1}(c_d),\qquad
q_{\mathrm{cmd}}=\bar q+s\rho_{\mathrm{FF}}
$$

である。feedback条件では、実測Tool DoF \(c_t\)を使い

$$
e_t=c_d-c_t,\qquad
\rho_{\mathrm{FB}}=f^{-1}(c_d)+K_pe_t
$$

とする。主比較では\(K_p=0.5\)、\(K_i=K_d=0\)を評価前に固定する。指令は\(\rho\)、\(\dot\rho\)、\(\ddot\rho\)、関節範囲で制限する。したがってfeedbackは目標角の真値を事後的に使うoracleではなく、環境のオンラインTool DoF観測を用いる制御器である。

同じtarget列と同じreset seedをFF/FBに与えるpaired設計とし、nominalと抵抗増加条件を分ける。抵抗増加は単一設定であるため、結論は「この抵抗設定における効果」に限定する。頑健性を主張する場合はdampingとtorqueの複数水準を掃引する追加実験を行う。

### 6.2 評価量

episode \(j\)、時刻\(t\)の誤差を\(e_{j,t}=c_{d,j}-c_{j,t}\)とする。一次指標はepisode RMSE

$$
\mathrm{RMSE}_j=\sqrt{\frac{1}{T_j}\sum_{t=1}^{T_j}e_{j,t}^2}
$$

である。副次指標はMAE、terminal error、\(|e|<0.01\) radの時間率、把持率、飽和率である。成功率や把持率は追従誤差と異なる機構を反映するため、RMSEだけで代用しない。

feedbackの効果は同一episodeの差

$$
\Delta_j=\mathrm{RMSE}^{\mathrm{FF}}_j-
          \mathrm{RMSE}^{\mathrm{FB}}_j
$$

で評価する。\(\Delta>0\)はfeedbackによる誤差低減を表す。episodeを再sampleするpaired bootstrapで\(\mathrm{mean}(\Delta)\)の95% CIを求め、CI下限が0より大きい場合だけ、当該task・当該外乱条件における低減の証拠とする。

現行の統合評価を実行するコードは以下である。これは固定方策から再収集するsmoke評価であり、正式DB主解析の代替ではない。

```bash
.venv/bin/python -m improved_tds.experiments.evaluate_scissors_tds \
  --algorithm sac \
  --models-root runs/scissors_sac_100epochs_20260713 \
  --output runs/scissors_tds_validation_20260713 \
  --calibration-episodes 100 --heldout-episodes 100 \
  --evaluation-episodes 30 --min-successes 20 \
  --kp 0.5 --bootstrap-samples 2000 --seed 12071200
```

## 7. 統計解析と採否規則

統計単位は**episode**であり、安定成功窓のstepを独立sampleとして数えない。相関はtest episodeの\((\rho_i,c_i)\)に対してPearson \(r\)とSpearman \(\rho_s\)を報告する。bootstrapではepisode indexを復元抽出する。permutation nullはTool DoFのepisode対応を置換した

$$
H_0:\rho\ \perp\ c
$$

の下の相関分布である。相関95% CIの下限がnullの95%上限を超えることを、task内の探索的な対応証拠として扱う。

5 task、2外乱条件、複数metricでは多重比較が生じる。主解析として事前に指定するのは、各taskのPearson相関と、nominal/resistanceのpaired RMSE差である。task横断でp値を解釈する場合はHolm補正を適用する。CIだけを報告する場合も、taskごとの推定でありfamily-wise確証ではないと明記する。

今回の5 task×各1 RL seedは**task内の方策条件付き評価**である。RL学習のばらつきまで含む結論には、taskごとに少なくとも3、望ましくは5の独立学習seedで、学習→3 split DB→TDS→制御評価を繰り返し、seedを上位の統計単位にする追加実験が必要である。

## 8. 論文用の図表と生成順序

|成果物|内容|主張との対応|
|---|---|---|
|表1: 実験条件|task、seed、環境、学習量、bin、quota、K、controller、外乱|再現性|
|図1: PC1–Tool DoF scatter|test episodeごとの\(c,\rho\)、回帰線、95% CI|表現の妥当性|
|表2: TDS fit/校正|EVR、再構成RMSE、Pearson/Spearman、校正RMSE、安全範囲|1次元モデルが成立する範囲|
|図2: FF/FB RMSE|task別・外乱別の平均とepisode bootstrap 95% CI|feedbackの寄与|
|図3: tracking trace|絶対誤差のmedian/IQR、0.01 rad閾値|誤差低減の時間構造|
|表3: 失敗・除外|成功率、quota未達bin、飽和率、把持率|選択バイアスと限界|
|付録図: 学習曲線|return、success、action飽和、checkpoint|方策品質と収束|
|付録図: gain/outlier感度|\(K_p\)と外乱水準の掃引|主結論の条件依存性|

`evaluate_scissors_tds`は`figure_pc1_tool_dof.{pdf,png}`、`figure_feedback_ablation.{pdf,png}`、`figure_tracking_error.{pdf,png}`、CSV、`metrics.json`、`validation_report.md`を出力する。提出用図はPDFを主とし、PNGは確認用にする。図のcaptionにはtask数、episode数、CI、統計単位、外乱条件、除外規則を記載する。

## 9. 報告時の議論テンプレート

結果の数値を得た後は、次の順で議論する。

1. **成立性**: quota達成率、成功率、EVRを先に示す。EVRが低い、または成功不足のtaskではTDS成立を主張しない。
2. **対応性**: 独立testのPearson/Spearmanとpermutation nullを示し、\(\rho\)と\(c\)の対応がtask依存であるかを述べる。
3. **制御性**: 校正RMSEと安全範囲を踏まえ、FFがどの目標範囲で追従できるかを述べる。
4. **feedback効果**: paired RMSE差のCI、把持率、飽和率を同時に示す。RMSE改善が把持低下や飽和増加と引き換えでないか確認する。
5. **限界**: 成功近傍への条件付け、単一RL seed、単一抵抗設定、シミュレータ内観測、Pゲイン固定を明記する。

「PC1との相関が高い」ことは「Tool DoFを制御できる」ことの必要条件の一つにすぎない。さらに「feedbackがRMSEを低減した」ことも、切断性能や実機安全性を自動的に意味しない。この三段階を分けて記述することで、TDS表現、逆較正、閉ループ制御に関する主張を過大化しない研究報告になる。

## 10. 提出前チェックリスト

- [ ] 全taskの`model_training_summary.json`が\(10^6\) stepsを記録している。
- [ ] `fit`、`calibration`、`test`の全summaryが`quota_met=true`である。
- [ ] PCA方向・符号はfit splitだけから決めた。
- [ ] 校正はcalibration splitだけでfitし、testに触れていない。
- [ ] 全CIの統計単位はepisodeである。
- [ ] FF/FBはtarget列・reset seed・安全制約が同一である。
- [ ] taskごとの成功率・除外数・飽和率を報告した。
- [ ] commit hash、依存version、XML、model SHA-256、全seed、CLI commandをartifactとして保存した。
- [ ] 主張を「評価済みtask・範囲・外乱・seed」に限定した。
