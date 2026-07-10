from __future__ import annotations

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
    ):
        super().__init__(env)
        self.output_path = None if output_path is None else Path(output_path)
        self.controller_name = controller_name
        self.policy_checkpoint = policy_checkpoint
        self.simulator_parameters = dict(simulator_parameters or {})
        self.save_failures = bool(save_failures)
        self.dataset = TrajectoryDataset()
        self._steps: list[TrajectoryStep] = []
        self._seed = 0
        self._success = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._steps = []
        self._seed = 0 if seed is None else int(seed)
        self._success = False
        return self.env.reset(seed=seed, options=options)

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        unwrapped = self.env.unwrapped
        required = ("get_joint_state", "get_tool_state", "get_tool_state_rate", "get_force_observation")
        if any(not hasattr(unwrapped, name) for name in required):
            raise TypeError("environment does not implement ArticulatedToolEnv data accessors")
        q, qd = unwrapped.get_joint_state()
        force = unwrapped.get_force_observation()
        self._success |= bool(float(np.asarray(info.get("is_success", 0.0)).reshape(-1)[0]))
        self._steps.append(
            TrajectoryStep(
                timestamp=float(getattr(getattr(unwrapped, "data", None), "time", len(self._steps))),
                q=q,
                qd=qd,
                action=np.asarray(action, dtype=np.float64).reshape(-1),
                tool_state=unwrapped.get_tool_state(),
                tool_state_rate=unwrapped.get_tool_state_rate(),
                joint_torque=force.get("joint_torque"),
                motor_current=force.get("motor_current"),
                contact_flags=force.get("contact_flags"),
                contact_forces=force.get("contact_forces"),
                phase=str(getattr(unwrapped, "phase", info.get("phase", "unknown"))),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        )
        if terminated or truncated:
            self._finish_episode()
        return obs, reward, terminated, truncated, info

    def _finish_episode(self) -> None:
        unwrapped = self.env.unwrapped
        metadata = TrajectoryMetadata(
            tool_family=str(getattr(unwrapped, "tool_family", "unknown")),
            tool_instance_id=str(getattr(unwrapped, "tool_instance_id", "unknown")),
            task_name=str(getattr(unwrapped, "task", getattr(self.env.spec, "id", "unknown"))),
            success=self._success,
            seed=self._seed,
            simulator_parameters=self.simulator_parameters,
            controller_name=self.controller_name,
            policy_checkpoint=self.policy_checkpoint,
            tool_state_unit="rad" if getattr(unwrapped, "joint_type", "hinge") == "hinge" else "m",
        )
        trajectory = Trajectory(metadata, list(self._steps))
        self.dataset.append(trajectory, successful_only=not self.save_failures)
        if self.output_path is not None and self.dataset.trajectories:
            self.dataset.save(self.output_path)
        self._steps = []

    def close(self) -> None:
        if self.output_path is not None and self.dataset.trajectories:
            self.dataset.save(self.output_path)
        super().close()

