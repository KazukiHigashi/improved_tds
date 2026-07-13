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
