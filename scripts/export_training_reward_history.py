"""Export auditable validation reward/success curves for the frozen 15-task RL run.

Each validation episode stores mean_reward_terms.reward_total as a mean over
that episode's simulator steps. We reconstruct the step-weighted mean at each
100k-step evaluation, not an exploratory training-rollout reward curve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "runs/paper15_unified_3m_slew_20260927"
OUTPUT = ROOT / "analysis/paper_ff_fb_generalization_20261008/training_reward_history.json"
TASKS = tuple(f"{category}_lite_{i}"
              for category in ("scissors", "push", "spray") for i in range(1, 6))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_task(task: str) -> dict:
    folder = TRAIN / task
    history_path = folder / "validation_history.json"
    summary_path = folder / "model_training_summary.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(history["poses"]) != 32 or len(history["targets"]) != 11:
        raise ValueError(f"unexpected validation design: {task}")
    points = []
    for row in history["history"]:
        episodes = row["episode_metrics"]
        if len(episodes) != 352 or row["episodes"] != 352:
            raise ValueError(f"incomplete validation checkpoint: {task}")
        total_steps = sum(int(ep["episode_steps"]) for ep in episodes)
        weighted_reward = sum(
            float(ep["mean_reward_terms"]["reward_total"]) * int(ep["episode_steps"])
            for ep in episodes)
        mean_step_reward = weighted_reward / total_steps
        if not math.isfinite(mean_step_reward):
            raise ValueError(f"nonfinite validation reward: {task}")
        points.append(dict(train_steps=int(row["timesteps"]),
                           validation_episodes=len(episodes),
                           validation_sim_steps=total_steps,
                           mean_reward_per_sim_step=mean_step_reward,
                           success_rate=float(row["success_rate"]),
                           mean_episode_return=weighted_reward / len(episodes)))
    if (not points or points[-1]["train_steps"] != summary["total_timesteps"]
            or any(b["train_steps"]-a["train_steps"] != 100_000
                   for a, b in zip(points, points[1:]))):
        raise ValueError(f"validation checkpoints do not match training: {task}")
    return dict(task=task, training_steps=summary["total_timesteps"],
                seed=int(summary["seed"]),
                target_range=[float(min(history["targets"])),
                              float(max(history["targets"]))],
                stopped_early=bool(summary["stopped_early"]),
                best_success_rate=float(summary["validation"]["best"]["success_rate"]),
                best_success_step=int(summary["validation"]["best"]["timesteps"]),
                history_sha256=digest(history_path), summary_sha256=digest(summary_path),
                points=points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest_path = TRAIN / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["protocol"] != "paper15_unified_3m_slew_direct_tds_v1"
            or manifest["tasks"] != list(TASKS)
            or manifest["training"]["validation_every_steps"] != 100_000):
        raise ValueError("training campaign differs")
    output = dict(protocol="paper15_validation_mean_step_reward_v1",
                  definition="sum(episode mean reward_total * episode steps) / sum(episode steps)",
                  source="validation_history.json, not training rollout",
                  manifest_sha256=digest(manifest_path),
                  tasks=[summarize_task(task) for task in TASKS])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    for row in output["tasks"]:
        print(row["task"], len(row["points"]), "validations",
              f"last reward/step={row['points'][-1]['mean_reward_per_sim_step']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
