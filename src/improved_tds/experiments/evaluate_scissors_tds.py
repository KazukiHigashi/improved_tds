from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import improved_tds
from improved_tds.control.base import ControlObservation
from improved_tds.control.feedback import FeedbackConfig, ToolStateFeedbackController
from improved_tds.control.feedforward import FeedforwardTDSController
from improved_tds.control.safety import SafetyLimiter, SafetyLimits
from improved_tds.environments.mujoco_utils import actuator_names, get_joint_qpos
from improved_tds.evaluation.scissors_tds import (
    correlation_with_episode_bootstrap,
    paired_bootstrap_difference,
)
from improved_tds.synergy.calibration import ToolStateCalibrator
from improved_tds.synergy.pca import PCATDS


@dataclass(frozen=True)
class PolicyEpisode:
    episode: int
    target: float
    tool_state: float
    q: np.ndarray
    best_angle_error: float
    grasped: bool
    successful: bool


def _actuated_q(unwrapped: Any) -> np.ndarray:
    values = []
    for name in actuator_names(unwrapped.model):
        joint_name = name.replace(":A_", ":")
        values.append(float(get_joint_qpos(unwrapped.model, unwrapped.data, joint_name)[0]))
    return np.asarray(values, dtype=np.float64)


def _normalized_action(unwrapped: Any, q_command: np.ndarray) -> np.ndarray:
    q = np.asarray(q_command, dtype=np.float64).reshape(-1)
    ctrlrange = np.asarray(unwrapped.model.actuator_ctrlrange, dtype=np.float64)
    if q.shape != (unwrapped.model.nu,):
        raise ValueError(f"q command has shape {q.shape}; expected {(unwrapped.model.nu,)}")
    center = np.mean(ctrlrange, axis=1)
    half_range = 0.5 * np.ptp(ctrlrange, axis=1)
    return np.clip((q - center) / half_range, -1.0, 1.0).astype(np.float32)


def _collect_policy_episodes(
    model_path: Path,
    *,
    algorithm: str,
    variant: int,
    episodes: int,
    seed: int,
) -> list[PolicyEpisode]:
    env = gym.make(f"ExpScissor{variant}-v0")
    from improved_tds.learning.algorithms import algorithm_class

    model = algorithm_class(algorithm).load(model_path, env=env, device="cpu")
    representatives: list[PolicyEpisode] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            candidates: list[tuple[float, bool, bool, float, np.ndarray]] = []
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                candidates.append(
                    (
                        float(info["angle_error"]),
                        bool(info["is_in_grasp_space"]),
                        bool(info["is_success"]),
                        float(info["tool_state"][0]),
                        _actuated_q(env.unwrapped),
                    )
                )
                done = bool(terminated or truncated)
            grasped_candidates = [item for item in candidates if item[1]]
            selected = min(grasped_candidates or candidates, key=lambda item: item[0])
            representatives.append(
                PolicyEpisode(
                    episode=episode,
                    target=float(observation["desired_goal"][0]),
                    tool_state=selected[3],
                    q=selected[4],
                    best_angle_error=selected[0],
                    grasped=selected[1],
                    successful=any(item[2] for item in candidates),
                )
            )
    finally:
        env.close()
    return representatives


def _successful_arrays(episodes: list[PolicyEpisode]) -> tuple[np.ndarray, np.ndarray]:
    selected = [episode for episode in episodes if episode.successful]
    if not selected:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.vstack([episode.q for episode in selected]), np.asarray(
        [episode.tool_state for episode in selected], dtype=np.float64
    )


def _make_controller(
    mode: str,
    estimator: PCATDS,
    calibrator: ToolStateCalibrator,
    unwrapped: Any,
    *,
    kp: float,
) -> FeedforwardTDSController:
    rho_low, rho_high = calibrator.safe_ranges()["rho"]
    span = max(rho_high - rho_low, 1e-3)
    ctrlrange = np.asarray(unwrapped.model.actuator_ctrlrange, dtype=np.float64)
    safety = SafetyLimiter(
        SafetyLimits(
            rho_min=rho_low,
            rho_max=rho_high,
            rho_rate_max=5.0 * span,
            rho_acceleration_max=50.0 * span,
            q_min=ctrlrange[:, 0],
            q_max=ctrlrange[:, 1],
        )
    )
    if mode == "feedback":
        return ToolStateFeedbackController(
            estimator,
            calibrator,
            safety,
            config=FeedbackConfig(kp=kp, ki=0.0, kd=0.0),
        )
    if mode == "feedforward":
        return FeedforwardTDSController(estimator, calibrator, safety)
    raise ValueError(f"unknown mode {mode!r}")


def _evaluate_controller(
    estimator: PCATDS,
    calibrator: ToolStateCalibrator,
    *,
    variant: int,
    targets: np.ndarray,
    condition: str,
    mode: str,
    seed: int,
    kp: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kwargs: dict[str, float] = {}
    if condition == "resistance":
        kwargs = {"hinge_damping": 0.2, "resistance_torque": 0.03}
    env = gym.make(f"ExpScissor{variant}-v0", **kwargs)
    unwrapped = env.unwrapped
    episode_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    try:
        for episode, target in enumerate(targets):
            env.reset(seed=seed + episode)
            unwrapped.goal = np.asarray([target], dtype=np.float32)
            controller = _make_controller(mode, estimator, calibrator, unwrapped, kp=kp)
            errors: list[float] = []
            grasp_values: list[float] = []
            success_values: list[float] = []
            saturated_values: list[float] = []
            for step in range(100):
                state = float(unwrapped.get_tool_state()[0])
                force = unwrapped.get_force_observation()
                output = controller.step(
                    float(target),
                    ControlObservation(
                        tool_state=state,
                        tool_state_rate=float(unwrapped.get_tool_state_rate()[0]),
                        synergy_force=float(force["synergy_force"]),
                    ),
                    unwrapped.dt,
                )
                action = _normalized_action(unwrapped, output.q_command)
                _, _, terminated, truncated, info = env.step(action)
                measured = float(info["tool_state"][0])
                error = abs(float(target) - measured)
                errors.append(error)
                grasp_values.append(float(info["is_in_grasp_space"]))
                success_values.append(float(info["is_success"]))
                saturated_values.append(float(output.saturated))
                trace_rows.append(
                    {
                        "variant": variant,
                        "condition": condition,
                        "mode": mode,
                        "episode": episode,
                        "step": step,
                        "time_s": step * unwrapped.dt,
                        "target_rad": float(target),
                        "tool_state_rad": measured,
                        "absolute_error_rad": error,
                        "grasped": int(bool(info["is_in_grasp_space"])),
                        "success": int(bool(info["is_success"])),
                    }
                )
                if terminated or truncated:
                    break
            values = np.asarray(errors, dtype=np.float64)
            episode_rows.append(
                {
                    "variant": variant,
                    "condition": condition,
                    "mode": mode,
                    "episode": episode,
                    "target_rad": float(target),
                    "mae_rad": float(np.mean(values)),
                    "rmse_rad": float(np.sqrt(np.mean(values**2))),
                    "terminal_error_rad": float(values[-1]),
                    "within_tolerance_fraction": float(np.mean(success_values)),
                    "grasp_fraction": float(np.mean(grasp_values)),
                    "saturation_fraction": float(np.mean(saturated_values)),
                }
            )
    finally:
        env.close()
    return episode_rows, trace_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + chr(10),
        encoding="utf-8",
    )


def _target_sequence(
    calibrator: ToolStateCalibrator, episodes: int, *, seed: int
) -> np.ndarray:
    low, high = calibrator.safe_ranges()["tool_state"]
    margin = 0.05 * (high - low)
    grid = np.linspace(low + margin, high - margin, max(min(episodes, 10), 2))
    values = np.resize(grid, episodes)
    return np.random.default_rng(seed).permutation(values)


def _plot_results(
    output: Path,
    correlation_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 5, figsize=(7.2, 1.8), constrained_layout=True)
    for variant, axis in enumerate(axes, start=1):
        rows = [row for row in correlation_rows if row["variant"] == variant]
        axis.set_title(f"Scissors {variant}")
        axis.set_xlabel("Tool DoF [rad]")
        if variant == 1:
            axis.set_ylabel("TDS PC1 score")
        if not rows:
            axis.text(
                0.5,
                0.5,
                "Not evaluable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            continue
        x = np.asarray([row["tool_state_rad"] for row in rows])
        y = np.asarray([row["pc1_score"] for row in rows])
        axis.scatter(x, y, s=8, alpha=0.65, edgecolors="none")
        slope, intercept = np.polyfit(x, y, 1)
        domain = np.linspace(x.min(), x.max(), 100)
        axis.plot(domain, intercept + slope * domain, color="black", lw=1)
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"figure_pc1_tool_dof.{suffix}", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), constrained_layout=True)
    positions = np.arange(1, 6)
    width = 0.32
    colors = {"feedforward": "#777777", "feedback": "#0072B2"}
    for condition_index, condition in enumerate(("nominal", "resistance")):
        axis = axes[condition_index]
        for mode_index, mode in enumerate(("feedforward", "feedback")):
            means = []
            error_low = []
            error_high = []
            for variant in range(1, 6):
                values = np.asarray(
                    [
                        row["rmse_rad"]
                        for row in episode_rows
                        if row["variant"] == variant
                        and row["condition"] == condition
                        and row["mode"] == mode
                    ],
                    dtype=np.float64,
                )
                if not values.size:
                    means.append(np.nan)
                    error_low.append(np.nan)
                    error_high.append(np.nan)
                    continue
                mean = float(np.mean(values))
                rng = np.random.default_rng(20_000 + 100 * variant + mode_index)
                bootstrap = np.asarray(
                    [
                        np.mean(rng.choice(values, size=values.size, replace=True))
                        for _ in range(1_000)
                    ]
                )
                low, high = np.quantile(bootstrap, [0.025, 0.975])
                means.append(mean)
                error_low.append(mean - low)
                error_high.append(high - mean)
            x = positions + (mode_index - 0.5) * width
            axis.bar(x, means, width, color=colors[mode], label=mode.capitalize())
            axis.errorbar(
                x,
                means,
                yerr=[error_low, error_high],
                fmt="none",
                color="black",
                lw=0.8,
            )
        axis.set_title("Nominal" if condition == "nominal" else "Increased resistance")
        axis.set_xlabel("Scissors variant")
        axis.set_xticks(positions)
        axis.set_ylabel("Tool DoF RMSE [rad]")
        axis.legend(frameon=False)
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"figure_feedback_ablation.{suffix}", bbox_inches="tight")
    plt.close(figure)

    eligible = sorted({row["variant"] for row in trace_rows})
    if not eligible:
        figure, axis = plt.subplots(figsize=(7.2, 2.4), constrained_layout=True)
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "No variant met the pre-registered TDS eligibility criterion.",
            ha="center",
            va="center",
        )
        for suffix in ("pdf", "png"):
            figure.savefig(output / f"figure_tracking_error.{suffix}", bbox_inches="tight")
        plt.close(figure)
        return
    figure, axes = plt.subplots(
        len(eligible),
        2,
        figsize=(7.2, max(1.6 * len(eligible), 2.4)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, variant in enumerate(eligible):
        for column_index, condition in enumerate(("nominal", "resistance")):
            axis = axes[row_index, column_index]
            for mode in ("feedforward", "feedback"):
                subset = [
                    row
                    for row in trace_rows
                    if row["variant"] == variant
                    and row["condition"] == condition
                    and row["mode"] == mode
                ]
                if not subset:
                    continue
                steps = sorted({int(row["step"]) for row in subset})
                matrix = np.asarray(
                    [
                        [row["absolute_error_rad"] for row in subset if row["step"] == step]
                        for step in steps
                    ]
                )
                median = np.median(matrix, axis=1)
                low, high = np.quantile(matrix, [0.25, 0.75], axis=1)
                time = np.asarray(steps) * 0.04
                axis.plot(time, median, color=colors[mode], label=mode.capitalize())
                axis.fill_between(time, low, high, color=colors[mode], alpha=0.18)
            axis.axhline(0.01, color="black", ls="--", lw=0.7)
            title = "Nominal" if condition == "nominal" else "Resistance"
            axis.set_title(f"S{variant}: {title}")
            axis.set_xlabel("Episode time [s]")
            axis.set_ylabel("Absolute error [rad]")
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False)
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"figure_tracking_error.{suffix}", bbox_inches="tight")
    plt.close(figure)


def _write_report(output: Path, metrics: dict[str, Any]) -> None:
    protocol = metrics["protocol"]
    lines = [
        "# ハサミ5条件におけるTDS-based tool control検証",
        "",
        "## 実験プロトコル",
        "",
        f"- PCA・較正用: 各条件{protocol['calibration_episodes']} episodes",
        f"- 独立相関評価用: 各条件{protocol['heldout_episodes']} episodes（別seed）",
        f"- 制御比較: 各条件・各controllerで{protocol['evaluation_episodes']} episodes",
        f"- TDS成立条件: 両splitで成功episodeが{protocol['min_successes']}件以上",
        f"- feedback gain: Kp={protocol['kp']:g}, Ki=0, Kd=0（評価前に固定）",
        "- mismatch条件: hinge damping=0.2、resistance torque=0.03",
        "- uncertainty: episode単位bootstrap 95% CI",
        "",
        "成功はTool DoF誤差0.01 rad未満かつ把持維持である。PC1の符号はPCA・較正用data",
        "だけで決め、相関は独立episodeで評価した。時刻stepを独立sampleとして扱っていない。",
        "",
        "## Tool DoFとTDS PC1",
        "",
        "| 条件 | 成立 | PCA成功数 | 評価成功数 | PC1説明分散比 | Pearson r [95% CI] | Spearman rho [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    variants = sorted(int(value) for value in metrics["variants"])
    for variant in variants:
        item = metrics["variants"][str(variant)]
        if not item["eligible"]:
            lines.append(
                f"| S{variant} | 否 | {item['calibration_successes']} | "
                f"{item['heldout_successes']} | — | — | — |"
            )
            continue
        correlation = item["correlation"]
        lines.append(
            f"| S{variant} | 可 | {item['calibration_successes']} | "
            f"{item['heldout_successes']} | {item['explained_variance_ratio']:.3f} | "
            f"{correlation['pearson_r']:.3f} "
            f"[{correlation['pearson_ci'][0]:.3f}, {correlation['pearson_ci'][1]:.3f}] | "
            f"{correlation['spearman_r']:.3f} "
            f"[{correlation['spearman_ci'][0]:.3f}, {correlation['spearman_ci'][1]:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## ツール状態フィードバックの効果",
            "",
            "正の誤差減少量はfeedbackがfeedforwardよりTool DoF RMSEを低減したことを示す。",
            "",
            "| 条件 | 環境 | Feedforward RMSE [rad] | Feedback RMSE [rad] | 平均誤差減少 [95% CI] |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for variant in variants:
        item = metrics["variants"][str(variant)]
        if not item["eligible"]:
            continue
        for condition in ("nominal", "resistance"):
            comparison = item["feedback_ablation"][condition]
            lines.append(
                f"| S{variant} | {condition} | {comparison['feedforward_rmse']:.4f} | "
                f"{comparison['feedback_rmse']:.4f} | "
                f"{comparison['mean_error_reduction']:.4f} "
                f"[{comparison['mean_95_ci'][0]:.4f}, "
                f"{comparison['mean_95_ci'][1]:.4f}] |"
            )
    eligible_items = [
        (variant, metrics["variants"][str(variant)])
        for variant in variants
        if metrics["variants"][str(variant)]["eligible"]
    ]
    supported_correlations = [
        variant
        for variant, item in eligible_items
        if item["correlation"]["pearson_ci"][0] > item["correlation"]["null_95"][1]
    ]
    feedback_supported: dict[str, list[int]] = {"nominal": [], "resistance": []}
    for variant, item in eligible_items:
        for condition in feedback_supported:
            if item["feedback_ablation"][condition]["mean_95_ci"][0] > 0.0:
                feedback_supported[condition].append(variant)
    lines.extend(
        [
            "",
            "## 議論",
            "",
            f"TDS成立条件を満たしたのは{len(eligible_items)}/{len(variants)}条件である。"
            "未達条件では、学習方策から課題関連TDSを同定できたとは結論しない。",
            f"独立dataでPC1相関の95% CIがpermutation null上限を上回った条件は"
            f"{supported_correlations or 'なし'}である。これは該当条件に限り、"
            "作動関節姿勢のPC1とTool DoFが対応する証拠となる。",
            f"feedbackによるRMSE低減の95% CIが0を上回った条件は、nominalで"
            f"{feedback_supported['nominal'] or 'なし'}、抵抗増加で"
            f"{feedback_supported['resistance'] or 'なし'}である。",
            "相関が強くても制御誤差が小さいとは限らない。PC1がTool DoFを表現できることと、"
            "較正逆写像による姿勢指令が接触力学下でTool DoFを実現できることは別の仮説である。",
        ]
    )
    lines.extend(
        [
            "",
            "## 解釈上の注意",
            "",
            "- 成立条件未達の環境では、失敗方策の主成分をTDSと解釈しない。",
            "- 同一target・同一seedのpaired比較であり、環境間の結果はpoolしていない。",
            "- 学習seedは1つだけであり、RL学習seed間の一般化は未検証である。",
            "- nominalと事前固定した抵抗増加条件以外への頑健性は主張しない。",
            "",
            "## 図",
            "",
            "- figure_pc1_tool_dof.pdf: 独立episodeのTool DoFとPC1 score",
            "- figure_feedback_ablation.pdf: feedforward/feedbackのTool DoF RMSE",
            "- figure_tracking_error.pdf: release後の絶対誤差中央値とIQR",
            "",
        ]
    )
    (output / "validation_report.md").write_text(chr(10).join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate TDS control on five scissor variants")
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--algorithm", choices=("sac", "td3", "ddpg"), default="sac")
    parser.add_argument("--variants", nargs="+", type=int, default=list(range(1, 6)))
    parser.add_argument("--calibration-episodes", type=int, default=100)
    parser.add_argument("--heldout-episodes", type=int, default=100)
    parser.add_argument("--evaluation-episodes", type=int, default=30)
    parser.add_argument("--min-successes", type=int, default=20)
    parser.add_argument("--kp", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=12_071_200)
    args = parser.parse_args(argv)
    counts = (
        args.calibration_episodes,
        args.heldout_episodes,
        args.evaluation_episodes,
        args.min_successes,
        args.bootstrap_samples,
    )
    if any(value <= 0 for value in counts):
        parser.error("episode counts, success threshold, and bootstrap count must be positive")
    if len(set(args.variants)) != len(args.variants) or not set(args.variants) <= set(range(1, 6)):
        parser.error("variants must be unique integers from 1 through 5")

    improved_tds.register_environments()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "protocol": {
            "calibration_episodes": args.calibration_episodes,
            "heldout_episodes": args.heldout_episodes,
            "evaluation_episodes": args.evaluation_episodes,
            "min_successes": args.min_successes,
            "kp": args.kp,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "variants": args.variants,
            "mismatch": {"hinge_damping": 0.2, "resistance_torque": 0.03},
        },
        "variants": {},
    }
    policy_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    for variant in args.variants:
        model_path = args.models_root / f"exp_scissor{variant}" / "model.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"missing trained model: {model_path}")
        calibration = _collect_policy_episodes(
            model_path,
            algorithm=args.algorithm,
            variant=variant,
            episodes=args.calibration_episodes,
            seed=args.seed + 10_000 * variant,
        )
        heldout = _collect_policy_episodes(
            model_path,
            algorithm=args.algorithm,
            variant=variant,
            episodes=args.heldout_episodes,
            seed=args.seed + 10_000 * variant + 5_000,
        )
        for split, rows in (("calibration", calibration), ("heldout", heldout)):
            for row in rows:
                policy_rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "episode": row.episode,
                        "target_rad": row.target,
                        "tool_state_rad": row.tool_state,
                        "best_angle_error_rad": row.best_angle_error,
                        "grasped": int(row.grasped),
                        "successful": int(row.successful),
                    }
                )
        calibration_q, calibration_state = _successful_arrays(calibration)
        heldout_q, heldout_state = _successful_arrays(heldout)
        item: dict[str, Any] = {
            "eligible": False,
            "calibration_successes": int(calibration_state.size),
            "heldout_successes": int(heldout_state.size),
            "calibration_success_rate": float(calibration_state.size / len(calibration)),
            "heldout_success_rate": float(heldout_state.size / len(heldout)),
        }
        metrics["variants"][str(variant)] = item
        if (
            calibration_state.size < args.min_successes
            or heldout_state.size < args.min_successes
        ):
            item["ineligible_reason"] = (
                f"成功episode不足: calibration={calibration_state.size}, "
                f"heldout={heldout_state.size}, threshold={args.min_successes}"
            )
            _json_dump(args.output / "metrics.json", metrics)
            continue

        estimator = PCATDS().fit(calibration_q, calibration_state)
        calibration_rho = estimator.encode(calibration_q)[:, 0]
        calibrator = ToolStateCalibrator(method="isotonic").fit(
            calibration_rho, calibration_state
        )
        heldout_rho = estimator.encode(heldout_q)[:, 0]
        correlation = correlation_with_episode_bootstrap(
            heldout_rho,
            heldout_state,
            seed=args.seed + variant,
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.bootstrap_samples,
        )
        reconstructed = estimator.decode(heldout_rho)
        item.update(
            {
                "eligible": True,
                "explained_variance_ratio": estimator.explained_variance_ratio_,
                "heldout_reconstruction_rmse": float(
                    np.sqrt(np.mean((reconstructed - heldout_q) ** 2))
                ),
                "calibrated_ranges": calibrator.safe_ranges(),
                "correlation": correlation.as_dict(),
            }
        )
        estimator.save(args.output / f"scissors_{variant}_pca_tds.npz")
        calibrator.save(args.output / f"scissors_{variant}_calibration.npz")
        for episode, (rho, state) in enumerate(zip(heldout_rho, heldout_state, strict=True)):
            correlation_rows.append(
                {
                    "variant": variant,
                    "episode": episode,
                    "tool_state_rad": float(state),
                    "pc1_score": float(rho),
                }
            )

        targets = _target_sequence(
            calibrator, args.evaluation_episodes, seed=args.seed + 100 * variant
        )
        item["feedback_ablation"] = {}
        for condition_index, condition in enumerate(("nominal", "resistance")):
            by_mode: dict[str, list[dict[str, Any]]] = {}
            for mode in ("feedforward", "feedback"):
                summaries, traces = _evaluate_controller(
                    estimator,
                    calibrator,
                    variant=variant,
                    targets=targets,
                    condition=condition,
                    mode=mode,
                    seed=args.seed + 1_000 * variant + 100 * condition_index,
                    kp=args.kp,
                )
                by_mode[mode] = summaries
                episode_rows.extend(summaries)
                trace_rows.extend(traces)
            feedforward_rmse = [row["rmse_rad"] for row in by_mode["feedforward"]]
            feedback_rmse = [row["rmse_rad"] for row in by_mode["feedback"]]
            comparison = paired_bootstrap_difference(
                feedforward_rmse,
                feedback_rmse,
                seed=args.seed + 10 * variant + condition_index,
                bootstrap_samples=args.bootstrap_samples,
            )
            comparison.update(
                {
                    "feedforward_rmse": float(np.mean(feedforward_rmse)),
                    "feedback_rmse": float(np.mean(feedback_rmse)),
                    "feedforward_success_fraction": float(
                        np.mean(
                            [
                                row["within_tolerance_fraction"]
                                for row in by_mode["feedforward"]
                            ]
                        )
                    ),
                    "feedback_success_fraction": float(
                        np.mean(
                            [
                                row["within_tolerance_fraction"]
                                for row in by_mode["feedback"]
                            ]
                        )
                    ),
                }
            )
            item["feedback_ablation"][condition] = comparison
        _json_dump(args.output / "metrics.json", metrics)

    _write_csv(args.output / "policy_episodes.csv", policy_rows)
    _write_csv(args.output / "pc1_heldout.csv", correlation_rows)
    _write_csv(args.output / "controller_episodes.csv", episode_rows)
    _write_csv(args.output / "controller_traces.csv", trace_rows)
    _json_dump(args.output / "metrics.json", metrics)
    _plot_results(args.output, correlation_rows, episode_rows, trace_rows)
    _write_report(args.output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
