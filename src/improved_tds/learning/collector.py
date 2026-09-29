from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from improved_tds.data.trajectory import (
    Trajectory,
    TrajectoryDataset,
    TrajectoryMetadata,
    TrajectoryStep,
)


class SuccessfulTrajectoryCollector(gym.Wrapper):
    """Capture synchronized full episodes and retain successful trajectories."""

    def __init__(
        self,
        env: gym.Env,
        *,
        output_path: str | Path | None = None,
        controller_name: str = "rl_policy",
        policy_checkpoint: str | None = None,
        simulator_parameters: dict[str, Any] | None = None,
        save_failures: bool = False,
        dataset_role: str = "training_candidate",
        save_every_trajectories: int = 100,
    ):
        super().__init__(env)
        self.output_path = None if output_path is None else Path(output_path)
        self.controller_name = controller_name
        self.policy_checkpoint = policy_checkpoint
        self.simulator_parameters = dict(simulator_parameters or {})
        self.save_failures = bool(save_failures)
        self.dataset_role = str(dataset_role)
        if save_every_trajectories < 1:
            raise ValueError("save_every_trajectories must be positive")
        self.save_every_trajectories = int(save_every_trajectories)
        self.dataset = TrajectoryDataset()
        self._steps: list[TrajectoryStep] = []
        self._seed: int | None = None
        self._success = False
        self._episode_id = -1
        self._episode_start_time = 0.0
        self._initial_desired_tool_state: list[float] | None = None
        self._previous_tool_state: np.ndarray | None = None
        self._initial_pose_index: int | None = None
        self._initial_pose_pool: str | None = None
        self.rejected_trajectories = 0
        self.rejection_reasons: dict[str, int] = {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._steps = []
        self._episode_id += 1
        self._seed = None if seed is None else int(seed)
        self._success = False
        obs, info = self.env.reset(seed=seed, options=options)
        data = getattr(self.env.unwrapped, "data", None)
        self._episode_start_time = float(getattr(data, "time", 0.0))
        desired = obs.get("desired_goal") if isinstance(obs, dict) else None
        self._initial_desired_tool_state = (
            None
            if desired is None
            else np.asarray(desired, dtype=np.float64).reshape(-1).tolist()
        )
        achieved = obs.get("achieved_goal") if isinstance(obs, dict) else None
        self._previous_tool_state = (
            None if achieved is None else np.asarray(achieved, dtype=np.float64).reshape(-1)
        )
        # Which authored pose the episode started from is a per-episode fact and
        # the confound the synergy analysis has to control for, so it travels
        # with the trajectory rather than being reconstructed afterwards.
        pose_index = info.get("initial_pose_index") if isinstance(info, dict) else None
        self._initial_pose_index = None if pose_index is None else int(pose_index)
        pose_pool = info.get("initial_pose_pool") if isinstance(info, dict) else None
        self._initial_pose_pool = None if pose_pool is None else str(pose_pool)
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get('numerical_failure',False):
            self.rejected_trajectories += 1
            self.rejection_reasons['numerical_failure'] = self.rejection_reasons.get('numerical_failure',0)+1
            self._success = False
            self._steps = []
            return obs,reward,terminated,truncated,info
        unwrapped = self.env.unwrapped
        q, qd, tool_state, tool_state_rate, force = self._read_state(unwrapped, obs)
        self._merge_reported_contacts(force, info)
        instantaneous_success = bool(
            float(np.asarray(info.get("is_success", 0.0)).reshape(-1)[0])
        )
        if "is_stable_success" in info:
            stable_success = bool(
                float(np.asarray(info["is_stable_success"]).reshape(-1)[0])
            )
        else:
            stable_success = instantaneous_success and bool(terminated)
        self._success |= stable_success
        desired = obs.get("desired_goal") if isinstance(obs, dict) else None
        data = getattr(unwrapped, "data", None)
        self._steps.append(
            TrajectoryStep(
                timestamp=float(getattr(data, "time", len(self._steps)))
                - self._episode_start_time,
                step_index=len(self._steps),
                q=q,
                qd=qd,
                action=np.asarray(action, dtype=np.float64).reshape(-1),
                tool_state=tool_state,
                desired_tool_state=(
                    None
                    if desired is None
                    else np.asarray(desired, dtype=np.float64).reshape(-1)
                ),
                tool_state_rate=tool_state_rate,
                joint_torque=force.get("joint_torque"),
                motor_current=force.get("motor_current"),
                contact_flags=force.get("contact_flags"),
                contact_forces=force.get("contact_forces"),
                phase=str(getattr(unwrapped, "phase", info.get("phase", "unknown"))),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                is_success=instantaneous_success,
                is_stable_success=stable_success,
                success_streak=int(info.get("success_streak", int(instantaneous_success))),
            )
        )
        if terminated or truncated:
            self._finish_episode()
        return obs, reward, terminated, truncated, info

    #: Contact channels recorded per step, in the order they are concatenated
    #: into ``contact_flags`` / ``contact_forces``. The first block is the digits
    #: holding the tool body; the second is the digits pressing the driven part.
    #: Keeping them apart is what lets the analysis tell a grasping digit from an
    #: operating one -- summed together they are indistinguishable.
    CONTACT_CHANNELS = ("grasp_digit", "tool_digit")

    @staticmethod
    def _merge_reported_contacts(force: dict[str, Any], info: dict[str, Any]) -> None:
        """Prefer per-digit contacts the environment computed over touch sensors.

        The tool-manip-gym models carry no MuJoCo sensors, so deriving the
        contact channels from ``sensordata`` yields empty arrays. Where the
        environment reports which digits touch the tool and how hard, that is
        the signal the grasp is witnessed by, so it takes precedence.
        """
        blocks = [
            (info.get("grasp_digit_contacts"), info.get("grasp_digit_normal_forces")),
            (
                info.get("trigger_digit_contacts", info.get("plunger_digit_contacts")),
                info.get(
                    "trigger_digit_normal_forces",
                    info.get("plunger_digit_normal_forces"),
                ),
            ),
        ]
        present = [pair for pair in blocks if pair[0] is not None and pair[1] is not None]
        if not present:
            return
        force["contact_flags"] = np.concatenate(
            [np.asarray(touching, dtype=bool).reshape(-1) for touching, _ in present]
        )
        force["contact_forces"] = np.concatenate(
            [np.asarray(forces, dtype=np.float64).reshape(-1) for _, forces in present]
        )

    def _read_state(self, unwrapped: gym.Env, obs: Any):
        """Read either the native improved-tds API or tool-manip-gym MuJoCo data."""
        required = ("get_joint_state", "get_tool_state", "get_tool_state_rate", "get_force_observation")
        if all(hasattr(unwrapped, name) for name in required):
            q, qd = unwrapped.get_joint_state()
            return (
                q,
                qd,
                unwrapped.get_tool_state(),
                unwrapped.get_tool_state_rate(),
                unwrapped.get_force_observation(),
            )
        data = getattr(unwrapped, "data", None)
        if data is None or not isinstance(obs, dict) or "achieved_goal" not in obs:
            raise TypeError("environment does not expose MuJoCo state or an achieved_goal")
        tool_state = np.asarray(obs["achieved_goal"], dtype=np.float64).reshape(-1)
        dt = float(getattr(unwrapped, "dt", 1.0))
        previous = self._previous_tool_state
        tool_state_rate = (
            np.zeros_like(tool_state)
            if previous is None or dt <= 0.0
            else (tool_state - previous) / dt
        )
        self._previous_tool_state = tool_state.copy()
        sensor_values = np.asarray(getattr(data, "sensordata", ()), dtype=np.float64).copy()
        force = {
            "joint_torque": np.asarray(data.actuator_force, dtype=np.float64).copy(),
            "contact_flags": sensor_values > 0.0,
            "contact_forces": np.clip(sensor_values, 0.0, None),
        }
        return (
            np.asarray(data.qpos, dtype=np.float64).copy(),
            np.asarray(data.qvel, dtype=np.float64).copy(),
            tool_state,
            tool_state_rate,
            force,
        )

    def _finish_episode(self) -> None:
        unwrapped = self.env.unwrapped
        simulator_parameters = dict(self.simulator_parameters)
        if self._initial_pose_index is not None:
            simulator_parameters["initial_pose_index"] = self._initial_pose_index
        if self._initial_pose_pool is not None:
            simulator_parameters["initial_pose_pool"] = self._initial_pose_pool
        metadata = TrajectoryMetadata(
            tool_family=str(getattr(unwrapped, "tool_family", "unknown")),
            tool_instance_id=str(getattr(unwrapped, "tool_instance_id", "unknown")),
            task_name=str(getattr(unwrapped, "task", getattr(self.env.spec, "id", "unknown"))),
            success=self._success,
            seed=self._seed,
            simulator_parameters=simulator_parameters,
            controller_name=self.controller_name,
            policy_checkpoint=self.policy_checkpoint,
            tool_state_unit="rad" if getattr(unwrapped, "joint_type", "hinge") == "hinge" else "m",
            episode_id=self._episode_id,
            stable_success_steps=int(getattr(unwrapped, "success_hold_steps", 1)),
            termination_reason=(
                "time_limit"
                if self._steps[-1].truncated
                else "terminated"
                if self._steps[-1].terminated
                else "unknown"
            ),
            dataset_role=self.dataset_role,
            initial_desired_tool_state=self._initial_desired_tool_state,
        )
        trajectory = Trajectory(metadata, list(self._steps))
        # Recording is bookkeeping alongside the run, not part of it. A rejected
        # trajectory used to abort training -- one malformed episode threw away
        # 835k steps of learning -- so the episode is dropped with a warning and
        # counted instead. `rejected_trajectories` is reported in the training
        # summary so the discarded episodes can be accounted for afterwards.
        try:
            self.dataset.append(trajectory, successful_only=not self.save_failures)
        except ValueError as exc:
            self.rejected_trajectories += 1
            self.rejection_reasons[str(exc)] = self.rejection_reasons.get(str(exc), 0) + 1
            warnings.warn(
                f"episode {self._episode_id} not recorded: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._steps = []
            return
        if (
            self.output_path is not None
            and self.dataset.trajectories
            and len(self.dataset.trajectories) % self.save_every_trajectories == 0
        ):
            self._save()
        self._steps = []

    @property
    def successful_trajectory_count(self) -> int:
        return len(self.dataset.successful)

    def _save(self) -> None:
        """Write the dataset, warning rather than failing if it cannot be written.

        Losing a checkpoint of the recording is not a reason to lose the run.
        """
        try:
            self.dataset.save(self.output_path)
        except Exception as exc:  # noqa: BLE001 - recording must never stop training
            self.rejected_trajectories += 0
            warnings.warn(f"trajectory dataset not saved: {exc}", RuntimeWarning, stacklevel=2)

    def close(self) -> None:
        if self.output_path is not None and self.dataset.trajectories:
            self._save()
        super().close()
