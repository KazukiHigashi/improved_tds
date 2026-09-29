"""Create a terminal-friendly Markdown companion to the HTML training report."""

from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "runs" / "tool_manip_1m"
TASKS = [
    *(f"scissors_lite_{i}" for i in range(1, 6)),
    *(f"slider_lite_{i}" for i in range(1, 6)),
]


def last_success_rate(log_path: Path) -> float:
    values = [float(line.rsplit("|", 2)[1].strip()) for line in log_path.read_text(encoding="utf-16").splitlines() if "success_rate" in line]
    return float(values[-1]) if values else 0.0


def main() -> None:
    lines = [
        "# tool-manip-gym 初期姿勢 pool 学習レポート",
        "",
        "## 概要",
        "",
        "- 学習済み: lite hand の scissors 5 task、slider 5 task",
        "- 学習量: 各 1,000,000 step、SAC + HER",
        "- 初期姿勢: 各 reset で `tool-pose-pipeline/grasps` の対応 pool からランダム選択",
        "- 画像付き物理ロールアウトは同じディレクトリの `training_report.html` に埋め込み済み",
        "",
        "## 総括",
        "",
        "| Task | 最終ログ success rate | モデル |",
        "| --- | ---: | --- |",
    ]
    for task in TASKS:
        rate = last_success_rate(RUN_ROOT / task / "train.log")
        lines.append(f"| {task} | {rate:.2f} | `{task}/model.zip` |")

    for task in TASKS:
        task_dir = RUN_ROOT / task
        summary = json.loads((task_dir / "model_training_summary.json").read_text())
        rate = last_success_rate(task_dir / "train.log")
        family = "scissors" if task.startswith("scissors") else "slider"
        lines.extend(
            [
                "",
                f"## {task}",
                "",
                "### 学習設定",
                "",
                f"- 環境: `{summary['env_id']}`",
                f"- seed: {summary['seed']}",
                f"- 学習 step: {summary['total_timesteps']:,}",
                f"- 初期姿勢 pool: `{Path(summary['pose_pool']).name}`",
                f"- 最終ログ success rate: {rate:.2f}",
                "",
                "### 保存モデルの物理ロールアウト",
                "",
                "保存モデルはランダム pool 初期姿勢から決定論的方策で最大 200 step 実行した。"
                "reset、50、100、200 step のレンダリングを `training_report.html` に埋め込み、"
                "手と工具の接触維持、工具の不自然な貫通・飛散、工具に触れずに成功する挙動がないかを確認した。",
                "",
                "### 考察",
                "",
            ]
        )
        if family == "scissors":
            lines.append(
                "学習時成功率は高い。ロールアウトでは把持を維持しながら工具角度を変化させる挙動を確認対象とした。"
                "単一 seed の可視化だけで報酬ハックがないと断定せず、複数 seed・目標角での定量評価を追加すべきである。"
            )
        else:
            lines.append(
                "slider の成功率は 0.08–0.32 と低い。主な候補は、初期把持接触の不安定さ、"
                "スライダの摩擦／接触剛性、目標報酬の希薄さである。改善として、接触安定度に応じた pose pool の重み付け、"
                "摩擦・ダンピングの校正、工具変位と接触維持の dense reward、段階的 curriculum が考えられる。"
            )

    lines.extend(
        [
            "",
            "## spray_lite_push",
            "",
            "未学習。環境観測が Dict 形式ではないため、Dict 観測を前提とする現在の SAC+HER replay buffer を初期化できなかった。"
            "HER を使わない SAC（`MlpPolicy` と通常 replay buffer）に切り替えてから、同じ初期姿勢 pool で再学習・物理評価する必要がある。",
            "",
            "## 留意点",
            "",
            "`stable_success_trajectories` は tool-manip-gym の終了条件と既存 collector の判定差により 0 になり得る。"
            "本レポートの主指標は SB3 ログの `success_rate` と、画像付きロールアウトによる物理挙動確認である。",
        ]
    )
    output = RUN_ROOT / "training_report.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
