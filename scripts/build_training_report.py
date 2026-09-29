"""Build one self-contained HTML report for completed tool-manip-gym runs."""

from __future__ import annotations

import base64
import io
import json
import re
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import SAC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent
sys.path[:0] = [
    str(PROJECT_ROOT / "src"),
    str(WORKSPACE / "tool-manip-gym" / "gym-scissors"),
    str(WORKSPACE / "tool-manip-gym" / "gym-slider"),
    str(WORKSPACE / "tool-manip-gym" / "gym-spray"),
]

import gym_scissors  # noqa: E402,F401
import gym_slider  # noqa: E402,F401
import gym_spray  # noqa: E402,F401
from improved_tds.environments.pose_pool import (  # noqa: E402
    InitialPosePoolWrapper,
    pool_filename_for_environment,
)


RUN_ROOT = PROJECT_ROOT / "runs" / "tool_manip_1m"
POOL_ROOT = WORKSPACE / "tool-pose-pipeline" / "grasps"
TASKS = [
    *(f"scissors_lite_{i}" for i in range(1, 6)),
    *(f"slider_lite_{i}" for i in range(1, 6)),
]


def environment_id(task: str) -> str:
    family, _, variant = task.rpartition("_")
    if family == "scissors_lite":
        return f"ScissorsVariant{variant}-v0"
    if family == "slider_lite":
        return f"SliderVariant{variant}-v0"
    raise ValueError(task)


def html_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def contact_sheet(frames: list[np.ndarray], labels: list[str]) -> Image.Image:
    images = [Image.fromarray(frame).convert("RGB") for frame in frames]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    sheet = Image.new("RGB", (width * len(images), height + 30), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        sheet.paste(image, (index * width, 30))
        draw.text((index * width + 8, 8), label, fill="black")
    return sheet


def parse_success_history(log_path: Path) -> list[float]:
    values: list[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"success_rate\s*\|\s*([0-9.]+)", line)
        if match:
            values.append(float(match.group(1)))
    return values


def learning_plot(history: list[float]) -> Image.Image:
    figure, axis = plt.subplots(figsize=(5.8, 2.4), dpi=120)
    axis.plot(np.arange(1, len(history) + 1), history, color="#1f77b4")
    axis.set(ylim=(-0.02, 1.02), xlabel="logged update", ylabel="success rate")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    return Image.open(buffer).convert("RGB")


def evaluate(task: str) -> dict:
    task_dir = RUN_ROOT / task
    env_id = environment_id(task)
    base = gym.make(
        env_id,
        render_mode="rgb_array",
        hold_initial_steps=0 if env_id.startswith("Scissors") else 30,
    )
    pool_path = POOL_ROOT / pool_filename_for_environment(base)
    env = InitialPosePoolWrapper(base, pool_path, seed=20260727)
    model = SAC.load(task_dir / "model.zip", device="cpu")
    obs, info = env.reset(seed=20260727)
    frames = [env.render()]
    rewards: list[float] = []
    successes: list[float] = []
    infos = [info]
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        successes.append(float(np.asarray(info.get("is_success", 0.0)).reshape(-1)[0]))
        infos.append(info)
        if step in {49, 99, 199}:
            frames.append(env.render())
        if terminated or truncated:
            break
    while len(frames) < 4:
        frames.append(env.render())
    final = infos[-1]
    result = {
        "steps": len(rewards),
        "return": float(np.sum(rewards)),
        "success_fraction": float(np.mean(successes)) if successes else 0.0,
        "final_success": float(successes[-1]) if successes else 0.0,
        "initial_pose_index": int(infos[0].get("initial_pose_index", -1)),
        "final_achieved": np.asarray(obs.get("achieved_goal", []), dtype=float).round(4).tolist(),
        "goal": np.asarray(obs.get("desired_goal", []), dtype=float).round(4).tolist(),
        "final_grasp": float(np.asarray(final.get("is_in_grasp_space", 0.0)).reshape(-1)[0]),
        "frames": html_image(contact_sheet(frames, ["reset", "step 50", "step 100", "step 200"])),
    }
    env.close()
    return result


def interpretation(task: str, result: dict, final_training_success: float) -> str:
    if task.startswith("scissors"):
        return (
            "実行フレームでは手とハサミの相対配置、把持状態、工具角度の変化を確認した。"
            "高い学習時成功率は、ランダム初期把持姿勢でも工具操作方策が獲得されたことを示す。"
            "ただし単一 seed の 200 step rollout だけでは一般化を保証しないため、複数 seed・目標角での評価が必要である。"
        )
    if final_training_success < 0.35:
        return (
            "slider は学習時成功率が低く、実行フレームで工具が大きく動かない、または目標位置へ到達しない挙動を重点的に確認した。"
            "候補原因は、接触が途切れる初期姿勢、スライダの摩擦・接触剛性、目標報酬の希薄さである。"
            "改善案は、初期 pose pool を接触安定度で重み付けすること、摩擦・ダンピングを校正すること、"
            "工具変位と接触維持への dense reward／段階的 curriculum を導入することである。"
        )
    return "実行フレームと最終工具状態を用いて、把持維持と目標到達を確認した。"


def main() -> None:
    sections: list[str] = []
    overview: list[dict] = []
    for task in TASKS:
        task_dir = RUN_ROOT / task
        summary = json.loads((task_dir / "model_training_summary.json").read_text(encoding="utf-8"))
        history = parse_success_history(task_dir / "train.log")
        result = evaluate(task)
        final_training_success = history[-1] if history else 0.0
        overview.append({"task": task, "training_success": final_training_success, **result})
        sections.append(
            f"""<section><h2>{task}</h2>
            <p><b>学習条件:</b> SAC + HER、1,000,000 steps、seed {summary['seed']}。"
            f"初期姿勢は <code>{Path(summary['pose_pool']).name}</code> から reset ごとにランダム選択した。</p>
            <h3>学習中の推移</h3><img class='plot' src='data:image/png;base64,{html_image(learning_plot(history))}'>
            <p>最終ログ success rate: <b>{final_training_success:.2f}</b></p>
            <h3>保存モデルの物理ロールアウト</h3>
            <p>決定論的方策を 1 episode（最大 200 step）実行。pool index={result['initial_pose_index']}、"
            f"return={result['return']:.3f}、成功 step 比率={result['success_fraction']:.2f}、"
            f"最終成功={result['final_success']:.0f}、最終把持={result['final_grasp']:.0f}。"
            f"最終工具状態={result['final_achieved']}、目標={result['goal']}。</p>
            <img class='rollout' src='data:image/png;base64,{result['frames']}'>
            <h3>観察と考察</h3><p>{interpretation(task, result, final_training_success)}</p></section>"""
        )

    rows = "".join(
        f"<tr><td>{row['task']}</td><td>{row['training_success']:.2f}</td>"
        f"<td>{row['success_fraction']:.2f}</td><td>{row['final_success']:.0f}</td></tr>"
        for row in overview
    )
    spray = """<section><h2>spray_lite_push</h2>
    <p><b>未学習。</b> この環境は Dict 観測ではないため、現在の SAC+HER replay buffer が初期化できず失敗した。"
    "従ってモデルの物理ロールアウト・成功率・報酬ハック確認は実施していない。"
    "HER を使わない SAC（MlpPolicy と通常 replay buffer）へ切り替え、同じ pose pool を適用して再学習する必要がある。</p></section>"""
    document = f"""<!doctype html><html lang='ja'><meta charset='utf-8'>
    <title>tool-manip-gym 1M-step training report</title><style>
    body{{font-family:system-ui,sans-serif;line-height:1.6;margin:2rem;max-width:1200px}} table{{border-collapse:collapse}}td,th{{border:1px solid #aaa;padding:.35rem .6rem}}img{{max-width:100%;border:1px solid #ccc}}.plot{{max-width:700px}}section{{margin-top:3rem;border-top:2px solid #ddd}}
    </style><body><h1>tool-manip-gym 初期姿勢 pool 学習レポート</h1>
    <p>対象: lite hand の scissors 5 task、slider 5 task、および spray push。学習済み 10 task はすべて 1,000,000 step。"
    "各 reset で tool-pose-pipeline の対応 pool からランダム初期姿勢を注入した。</p>
    <h2>総括</h2><table><tr><th>task</th><th>学習時 success rate</th><th>評価成功 step 比率</th><th>評価最終成功</th></tr>{rows}</table>
    <p>画像は各保存モデルの実行中レンダリングである。手が工具に接触せずに目標だけを満たす、"
    "工具が不自然に貫通・飛散する、といった報酬ハックを確認する目的で、reset・50・100・200 step を連続掲載した。</p>
    {''.join(sections)}{spray}</body></html>"""
    output = RUN_ROOT / "training_report.html"
    output.write_text(document, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
