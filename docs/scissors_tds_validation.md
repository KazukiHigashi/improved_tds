# ハサミTDS-based tool control検証手順

> 旧DDPG学習runは、episode内holdによる遷移不整合と姿勢DB未保存が判明したため正式評価へ使用しない。本手順は、修正環境で再学習し、固定checkpointから`formal_balanced` DBを作成した後に適用する。

## 目的

5種類のハサミについて、次の仮説を環境ごとに検証する。

1. 作動関節姿勢のTDS PC1 scoreがTool DoF（ハサミhinge角）と対応する。
2. TDS逆較正による姿勢指令でTool DoFを制御できる。
3. tool-state feedbackがfeedforwardだけの場合よりTool DoF誤差を低減する。

## 成立条件とdata分割

学習済みSAC/TD3/DDPG+HER方策を決定論的にrolloutし、各episodeから、把持中に角度誤差が最小だった
作動関節姿勢を1 sampleだけ抽出する。成功は角度誤差0.01 rad未満かつ把持維持とする。
PCA・符号整合・単調較正用と相関評価用には異なるseedを用いる。両splitで成功episodeが
20件以上得られない環境はTDS成立条件未達とし、失敗軌道のPC1をTDSとして解釈しない。

相関は独立評価episodeのPearson相関とSpearman順位相関を使用する。95%信頼区間は
episode単位bootstrapで求め、時刻stepを独立sample数として数えない。Tool DoFをepisode間で
permutationしたnull相関の95%区間も併記する。

## フィードバック比較

各環境で、同一targetと同一reset seedを用いて次をpaired比較する。

- feedforward: rho_cmd = f^-1(c_d)
- feedback: rho_cmd = f^-1(c_d) + Kp(c_d-c)

Kp=0.5、Ki=0、Kd=0は評価前に固定し、評価dataによる調整は行わない。nominal条件に加え、
hinge damping=0.2、resistance torque=0.03の抵抗増加条件を事前に固定して評価する。
主指標はrelease後のTool DoF RMSEとし、誤差0.01 rad以内の時間率、把持率、飽和率も保存する。

## 実行

5モデルのexit code、candidate DB、formal balanced DBがすべて有効であることを確認した後、次を実行する。

    env MUJOCO_GL=egl OMP_NUM_THREADS=1 .venv/bin/python -m improved_tds.experiments.evaluate_scissors_tds --algorithm sac --models-root runs/scissors_sac_100epochs_20260713 --output runs/scissors_tds_validation

出力には、生dataのCSV、集計JSON、PCA-TDSと較正model、PDF/PNG図、日本語の
validation_report.mdが含まれる。
