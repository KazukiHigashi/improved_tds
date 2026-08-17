# Improved TDS

`improved-tds`は、既知または初期化済みの把持で1-DoF関節ツールを低次元閉ループ制御する独立研究projectである。参照元`tool-dof-synergy`を変更せず、PCA/feedforward baselineを発展させる。

version付き成功軌道dataset、PCA・教師あり・family TDS推定器、単調few-shot較正、feedforward/PID/admittance controller、独立安定化synergy、力・hardware interface、および固定されたGymnasium/MuJoCo評価環境を実装している。

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

数値coreはCPUだけで動作し、Gymnasium、MuJoCo、SB3、GPUに依存しない。環境には`.[simulation]`、任意のSB3 SAC/TD3/DDPG+HER data生成baselineには`.[training]`を導入する。Isaac Simなどの重量級simulatorは使用しない。

## 互換性と環境

コピーしたMIT licenseのハサミassetはproject内で完結する。`ExpScissor1-v0`から`ExpScissor5-v0`は、HER用Dict観測と正規化14-actuator actionの契約を維持する。追加環境IDは`TDS-Trigger-v0`と`TDS-Button-v0`である。

### Gymnasium環境のスコープ

このリポジトリでは、既存Gymnasium環境をTDS研究の固定評価基盤として扱い、後方互換性の維持、bug修正、再現性に必要な最小限の変更だけを行う。新task、報酬設計、観測・action契約、asset、物理モデルなどの環境機能の拡充は、このリポジトリでは扱わない。環境自体の拡充は専用の環境リポジトリで実施し、安定版を依存関係または明示的なversionとして取り込む。

## 初期姿勢DBの対応

`tool-manip-gym`を用いるSB3学習では、`InitialPosePoolWrapper`が各episodeのreset時に
対応するNPZから1姿勢をランダムに選び、hand actuator姿勢とhand mount変位を適用する。
初期姿勢自体を学習する経路や、pose poolがない場合のfallbackは使用しない。

`train_sb3`の既定ディレクトリは、workspace直下の
`tool-pose-pipeline/grasps/`である。ファイル名は環境のtask、hand、variantから
`{task}_{hand}_{variant}.npz`として決まり、NPZ内のtask、hand、variant、actuator名も
実環境と一致しなければならない。

### 標準の対応表

| task | 主な環境ID | 標準の初期姿勢DB |
| --- | --- | --- |
| Lite scissors 1–5 | `ShadowHandLiteScissors{1-5}-v0`, `ScissorsVariant{1-5}-v0` | `tool-pose-pipeline/grasps/scissors_lite_{1-5}.npz` |
| Shadow scissors 1–5 | `ShadowHandScissors{1-5}-v0` | `tool-pose-pipeline/grasps/scissors_shadow_{1-5}.npz` |
| Lite slider 1–5 | `ShadowHandLiteSlider{1-5}-v0`, `SliderVariant{1-5}-v0` | `tool-pose-pipeline/grasps/slider_lite_{1-5}.npz` |
| Shadow slider 1–5 | `ShadowHandSlider{1-5}-v0` | `tool-pose-pipeline/grasps/slider_shadow_{1-5}.npz` |
| Lite push bottle 1–5 | `ShadowHandLitePush{1-5}-v0` | `tool-pose-pipeline/grasps/push_lite_{1-5}.npz` |
| Lite trigger spray 1–5 | `ShadowHandLiteSpray{1-5}-v0` | `tool-pose-pipeline/grasps/spray_lite_spray{1-5}.npz` |
| 旧pump式spray（Lite） | `ShadowHandLitePushSpray-v0` | `tool-pose-pipeline/grasps/spray_lite_push.npz` |
| 旧pump式spray（Shadow） | `ShadowHandPushSpray-v0` | `tool-pose-pipeline/grasps/spray_shadow_push.npz` |

現時点では、Shadow版push bottle 1–5に対応する`push_shadow_{1-5}.npz`と、標準
`grasps/`内のShadow版trigger spray poolは用意されていない。該当環境を学習する場合は、
先に対応するpoolを生成・検証すること。`ShadowHandLiteSpray{1-5}Pose-v0`はeditor出力を
直接確認するための環境であり、pose poolを注入する学習には使用しない。

### version付きpoolと実行済みrun

`grasps/`が標準である一方、sprayの改善実験ではversion付きpoolを明示指定している。
モデル評価やシナジー解析では、必ず学習時と同じpoolを使う。

| run / 用途 | 使用した初期姿勢DB |
| --- | --- |
| `runs/tool_manip_1m` | scissors/slider Liteと旧pump式sprayに`grasps/` |
| `runs/tool_manip_1m_10pos` | `grasps-10pos/`。過去の10姿勢実験用 |
| `runs/push_lite_dense_reward_1m_20260811` | push1–5に`grasps/push_lite_{1-5}.npz` |
| `runs/spray_lite_1m_reward_v2_20260814` | spray1–5に`grasps/spray_lite_spray{1-5}.npz` |
| `runs/spray_lite_1m_reward_v3_20260815` | spray1–5に`grasps/spray_lite_spray{1-5}.npz` |
| `runs/spray_lite_1m_convex_v4_20260815` | spray2–5に`grasps_v2/spray_lite_spray{2-5}.npz`。`grasps_v2`にはspray1がない |
| `runs/spray_lite_1m_graspfix_v5_20260816` | spray1–5に`grasps_v3/spray_lite_spray{1-5}.npz` |
| v5 checkpoint screening・シナジー収集・TDS評価 | 学習時と同じ`grasps_v3/` |

最新のspray v5系解析では`grasps_v3/`が基準である。ただし
`scripts/train_spray_lite_tasks.ps1`の`-PoolDir`既定値は現在も
`tool-pose-pipeline\grasps`なので、v5条件を再現するときは明示的に指定する。

```powershell
.\scripts\train_spray_lite_tasks.ps1 `
  -PoolDir "tool-pose-pipeline\grasps_v3" `
  -RunName "spray_lite_1m_graspfix_v5_reproduction"
```

実行済みモデルが使用したpoolの最終的なsource of truthは、各task directoryの
`model_training_summary.json`に保存された`pose_pool`である。directory名やREADMEの
記載だけから推測せず、再評価前にこの値を確認する。

### 外部pose poolを使わない環境

`TDS-Trigger-v0`、`TDS-Button-v0`、およびこのリポジトリ内蔵の
`ExpScissor{1-5}-v0`を通常の収集・評価CLIで使う場合、初期状態は各環境実装・MJCF側で
定義され、上記NPZは注入されない。これらと、`tool-manip-gym`環境を
`train_sb3`で学習する経路を区別すること。

```python
import gymnasium as gym
import improved_tds

env = gym.make("TDS-Trigger-v0")
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

end-to-endのdata収集・fit commandは[`docs/experiments.md`](docs/experiments.md)、architecture調査と互換性判断は[`docs/implementation_plan_closed_loop_tds.md`](docs/implementation_plan_closed_loop_tds.md)に記載する。

## 安全境界

simulation既定値は実機の安全制限ではない。motor定数、sensor較正、command transport、安全制限は推測で補わない。実機統合では`HardwareAdapter`を実装し、検証済み設定と[`docs/real_robot_integration.md`](docs/real_robot_integration.md)の緊急解放・watchdog動作を維持する。

ハサミassetのupstream noticeは[`src/improved_tds/assets/LICENSE.md`](src/improved_tds/assets/LICENSE.md)に保持する。
