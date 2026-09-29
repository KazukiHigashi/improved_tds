"""Deterministic validation and early stopping for SB3 tool policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from improved_tds.evaluation.reports import write_report


def score_policy_on_grid(
    env,
    policy,
    poses: Iterable[int],
    targets: Iterable[float],
    option_key: str,
) -> dict[str, Any]:
    """Evaluate one deterministic policy on every pose-target combination."""
    pose_values = tuple(int(pose) for pose in poses)
    target_values = tuple(float(target) for target in targets)
    successes = 0
    errors: list[float] = []
    target_metrics: list[dict[str, Any]] = []
    final_successes = 0
    precision_successes = 0
    grasp_steps = total_steps = 0
    raw = getattr(env, 'unwrapped', env)
    if hasattr(raw, 'target_angle_range'):
        span = float(np.diff(raw.target_angle_range)[0])
    elif hasattr(raw, 'push_limit'):
        span = float(raw.push_limit)
    elif hasattr(raw, 'trigger_limit'):
        span = float(raw.trigger_limit)
    else:
        span = max(max(target_values) - min(target_values), 1e-12)
    tolerance_counts = {str(t): 0 for t in (0.01, 0.03, 0.05, 0.075)}
    episode_metrics = []
    for target in target_values:
        target_successes = 0
        target_errors: list[float] = []
        for pose in pose_values:
            observation, reset_info = env.reset(
                seed=pose,
                options={"initial_pose_index": pose, option_key: target},
            )
            succeeded = False
            achieved = np.nan
            streaks = {t: 0 for t in tolerance_counts}
            attained = {t: False for t in tolerance_counts}
            held_after_success = 0
            precision_streak = 0
            precision_attained = False
            min_angle, max_angle = np.inf, -np.inf
            reward_sums = {}
            episode_steps = saturated = 0
            slow_steps = near_steps = 0
            speed_sum = 0.0
            max_hold = current_hold = 0
            frozen_frames = freeze_attempts = freeze_aborts = max_fixed_hold = 0
            previous_frozen = False
            hold_break_recorded = False
            hold_break_causes = dict(angle=0, speed=0, support=0, physical=0)
            servo_gap_sum = servo_gap_max = 0.0
            servo_gap_frames = 0
            while True:
                action, _ = policy.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                episode_steps += 1
                frozen = bool(info.get('retention_frozen',False))
                if frozen and not previous_frozen:
                    hold_break_recorded = False
                if frozen and not info.get('is_success', False) and not hold_break_recorded:
                    hold_break_causes['angle'] += int(float(info.get('angle_error', 0.)) > .03)
                    hold_break_causes['speed'] += int(float(info.get('tool_speed', 0.)) > .1)
                    hold_break_causes['support'] += int(not info.get('grasp_maintained', True))
                    hold_break_causes['physical'] += int(info.get('task_failure', False))
                    hold_break_recorded = True
                gap = info.get('servo_gap_max_rad')
                if gap is not None and np.isfinite(gap) and not info.get('numerical_failure', False):
                    servo_gap_sum += float(gap)
                    servo_gap_max = max(servo_gap_max, float(gap))
                    servo_gap_frames += 1
                frozen_frames += int(frozen)
                freeze_attempts += int(frozen and not previous_frozen)
                freeze_aborts += int(info.get('retention_aborted',False))
                max_fixed_hold = max(max_fixed_hold,int(info.get('retention_success_steps',0)))
                previous_frozen = frozen
                saturated += float(np.mean(np.abs(np.asarray(action)) >= 0.95))
                for name, value in info.items():
                    if name.startswith('reward_') and np.isscalar(value):
                        reward_sums[name] = reward_sums.get(name, 0.0) + float(value)
                success = info.get("is_stable_success", info.get("is_success", 0.0))
                succeeded = succeeded or bool(success)
                if isinstance(observation, dict) and "achieved_goal" in observation:
                    achieved = float(
                        np.asarray(observation["achieved_goal"]).reshape(-1)[0]
                    )
                elif "achieved_goal" in info:
                    achieved = float(np.asarray(info["achieved_goal"]).reshape(-1)[0])
                grasped = bool(info.get("grasp_maintained", True))
                slow = float(info.get("tool_speed", 0.0)) <= 0.10
                slow_steps += int(slow)
                near_steps += int(abs(achieved-target) <= .03*span)
                speed_sum += float(info.get('tool_speed',0.0))
                grasp_steps += int(grasped)
                total_steps += 1
                for tolerance in streaks:
                    good = grasped and (slow or info.get('speed_success_criterion') is False) and abs(achieved-target) <= float(tolerance)*span
                    streaks[tolerance] = streaks[tolerance] + 1 if good else 0
                    attained[tolerance] |= streaks[tolerance] >= int(info.get("success_hold_steps", 5))
                precision_streak = precision_streak + 1 if info.get("precision_success", False) else 0
                precision_attained |= precision_streak >= int(info.get("success_hold_steps", 5))
                held_after_success += int(succeeded and bool(info.get("is_success", False)))
                current_hold = current_hold + 1 if info.get('is_success',False) else 0
                max_hold = max(max_hold,current_hold)
                if grasped:
                    min_angle, max_angle = min(min_angle, achieved), max(max_angle, achieved)
                if terminated or truncated:
                    break
            if info.get('numerical_failure',False):
                succeeded = False
                precision_attained = False
                attained = {t:False for t in attained}
            final_successes += int(bool(info.get("is_stable_success", info.get("is_success", False))))
            precision_successes += int(precision_attained)
            for tolerance in tolerance_counts:
                tolerance_counts[tolerance] += int(attained[tolerance])
            episode_metrics.append(dict(pose_index=pose, target=target, success=succeeded,
                final_error=None if info.get('numerical_failure',False) else abs(achieved-target), supported_min=None if not np.isfinite(min_angle) else min_angle,
                supported_max=None if not np.isfinite(max_angle) else max_angle,
                successful_steps_after_first_hold=held_after_success,
                max_consecutive_hold_steps=max_hold,
                action_saturation_fraction=saturated/max(episode_steps,1),
                mean_reward_terms={k:v/max(episode_steps,1) for k,v in reward_sums.items()},
                task_failure=bool(info.get('task_failure',False)),
                episode_steps=episode_steps,
                mean_tool_speed=speed_sum/max(episode_steps,1),
                slow_fraction=slow_steps/max(episode_steps,1),
                near_target_fraction=near_steps/max(episode_steps,1),
                failure_contact=bool(info.get('failure_contact',False)),
                failure_displacement=bool(info.get('failure_displacement',False)),
                failure_rotation=bool(info.get('failure_rotation',False)),
                object_dropped=bool(info.get('object_dropped',False)),
                final_support_bodies=info.get('grasp_contact_count'),
                initial_supported=bool(reset_info.get('grasp_maintained',False)),
                frozen_steps=frozen_frames,freeze_attempts=freeze_attempts,freeze_aborts=freeze_aborts,
                max_fixed_hold_steps=max_fixed_hold,numerical_failure=bool(info.get('numerical_failure',False)),
                hold_break_causes=hold_break_causes,
                mean_servo_gap_rad=servo_gap_sum/servo_gap_frames if servo_gap_frames else None,
                max_servo_gap_rad=servo_gap_max if servo_gap_frames else None,
                physical_failure_after_success=bool(succeeded and info.get('task_failure',False))))
            successes += int(succeeded)
            target_successes += int(succeeded)
            if not info.get('numerical_failure',False):
                target_errors.append(abs(achieved - target))
        errors.extend(target_errors)
        target_metrics.append(
            {
                "target": target,
                "episodes": len(pose_values),
                "successes": target_successes,
                "success_rate": target_successes / len(pose_values),
                "median_abs_error": float(np.median(target_errors)) if target_errors else None,
                "mean_abs_error": float(np.mean(target_errors)) if target_errors else None,
                "error_observations": len(target_errors),
            }
        )
    episodes = len(pose_values) * len(target_values)
    per_target_rates = [float(item["success_rate"]) for item in target_metrics]
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "median_abs_error": float(np.median(errors)) if errors else None,
        "mean_abs_error": float(np.mean(errors)) if errors else None,
        "error_observations": len(errors),
        "numerical_failure_episodes": sum(x['numerical_failure'] for x in episode_metrics),
        "min_target_success_rate": min(per_target_rates),
        "target_coverage_fraction": float(np.mean(np.asarray(per_target_rates) > 0.0)),
        "target_metrics": target_metrics,
        "final_success_rate": final_successes / episodes,
        "precision_success_rate": precision_successes / episodes,
        "grasp_maintenance_rate": grasp_steps / max(total_steps, 1),
        "normalized_mean_abs_error": float(np.mean(errors)) / span if errors else None,
        "tolerance_success_rates": {t: n/episodes for t,n in tolerance_counts.items()},
        "episode_metrics": episode_metrics,
    }


def ranking_error(result):
    value = result.get('median_abs_error')
    return float(value) if value is not None else float('inf')


def validation_result_is_better(
    candidate: dict[str, Any], best: dict[str, Any] | None
) -> bool:
    """Prefer range coverage before aggregate success and tracking error."""
    if best is None:
        return True
    candidate_key = (
        float(candidate.get("min_target_success_rate", candidate["success_rate"])),
        float(candidate.get("target_coverage_fraction", 1.0)),
        float(candidate["success_rate"]),
        -ranking_error(candidate),
    )
    best_key = (
        float(best.get("min_target_success_rate", best["success_rate"])),
        float(best.get("target_coverage_fraction", 1.0)),
        float(best["success_rate"]),
        -ranking_error(best),
    )
    return candidate_key > best_key


class ValidationEarlyStoppingCallback(BaseCallback):
    """Validate periodically, save the best policy, and stop on a success streak."""

    def __init__(
        self,
        *,
        validation_env,
        poses: Iterable[int],
        targets: Iterable[float],
        option_key: str,
        eval_freq: int,
        success_threshold: float,
        consecutive_evaluations: int,
        best_policy_path: str | Path,
        report_path: str | Path,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.validation_env = validation_env
        self.poses = tuple(int(pose) for pose in poses)
        self.targets = tuple(float(target) for target in targets)
        self.option_key = option_key
        self.eval_freq = int(eval_freq)
        self.success_threshold = float(success_threshold)
        self.consecutive_evaluations = int(consecutive_evaluations)
        self.best_policy_path = Path(best_policy_path)
        self.report_path = Path(report_path)
        self.history: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.best_coverage: dict[str, Any] | None = None
        self.success_streak = 0
        self.stopped_early = False
        self._next_evaluation = self.eval_freq

    def _report(self) -> dict[str, Any]:
        return {
            "eval_freq": self.eval_freq,
            "success_threshold": self.success_threshold,
            "consecutive_evaluations": self.consecutive_evaluations,
            "poses": list(self.poses),
            "targets": list(self.targets),
            "option_key": self.option_key,
            "best_policy": (
                str(self.best_policy_path.with_suffix(".zip")) if self.best else None
            ),
            "best": self.best,
            "best_pose_coverage": self.best_coverage,
            "best_success_policy": str(self.best_policy_path.parent / "best_success_policy.zip"),
            "best_pose_coverage_policy": str(self.best_policy_path.parent / "best_pose_coverage_policy.zip"),
            "success_streak": self.success_streak,
            "stopped_early": self.stopped_early,
            "history": self.history,
        }

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_evaluation:
            return True
        result = score_policy_on_grid(
            self.validation_env,
            self.model,
            self.poses,
            self.targets,
            self.option_key,
        )
        result["timesteps"] = int(self.num_timesteps)
        self.history.append(result)

        if validation_result_is_better(result, self.best_coverage):
            self.best_coverage = dict(result)
            self.best_policy_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.best_policy_path.parent / "best_pose_coverage_policy")

        if self.best is None or (result['success_rate'], -ranking_error(result)) > (self.best['success_rate'], -ranking_error(self.best)):
            self.best = dict(result)
            self.best_policy_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.best_policy_path)
            self.model.save(self.best_policy_path.parent / "best_success_policy")

        if self.option_key == 'target_angle' and hasattr(self.validation_env, "goal_sampling_report"):
            result['training_goal_sampling'] = self.training_env.env_method('goal_sampling_report')

        if result["success_rate"] >= self.success_threshold:
            self.success_streak += 1
        else:
            self.success_streak = 0
        self.stopped_early = self.success_streak >= self.consecutive_evaluations
        write_report(self.report_path, self._report())

        self.logger.record("validation/success_rate", result["success_rate"])
        self.logger.record(
            "validation/min_target_success_rate",
            result.get("min_target_success_rate", result["success_rate"]),
        )
        self.logger.record(
            "validation/target_coverage_fraction",
            result.get("target_coverage_fraction", 1.0),
        )
        self.logger.record("validation/median_abs_error", result["median_abs_error"])
        self.logger.record("validation/consecutive_successes", self.success_streak)
        if self.verbose:
            print(
                "validation at "
                f"{self.num_timesteps} steps: {result['successes']}/"
                f"{result['episodes']} ({result['success_rate']:.1%}), "
                f"streak {self.success_streak}/{self.consecutive_evaluations}",
                flush=True,
            )
            if self.stopped_early:
                print("early stopping criterion reached", flush=True)
        self._next_evaluation += self.eval_freq
        return not self.stopped_early

    def _on_training_end(self) -> None:
        write_report(self.report_path, self._report())
        self.validation_env.close()
