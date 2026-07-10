from __future__ import annotations

from dataclasses import dataclass

from improved_tds.environments.one_dof import OneDoFToolEnv


@dataclass(frozen=True)
class TriggerParameters:
    travel: float = 0.35
    spring_stiffness: float = 1.0
    damping: float = 0.05
    preload: float = 0.0
    friction: float = 0.01
    lever_arm: float = 0.04
    actuation_threshold: float = 0.25
    body_mass: float = 0.05
    handle_friction: float = 0.5
    force_limit: float = 2.0


class TriggerEnv(OneDoFToolEnv):
    def __init__(
        self,
        parameters: TriggerParameters | None = None,
        *,
        instance_id: str = "trigger-default",
        task: str = "target_displacement",
        target_range: tuple[float, float] | None = None,
        max_episode_steps: int = 200,
        n_substeps: int = 10,
        render_mode: str | None = None,
    ):
        self.parameters = parameters or TriggerParameters()
        p = self.parameters
        super().__init__(
            family="trigger",
            instance_id=instance_id,
            joint_type="hinge",
            travel=p.travel,
            stiffness=p.spring_stiffness,
            damping=p.damping,
            friction=p.friction,
            preload=p.preload,
            mass=p.body_mass,
            force_limit=p.force_limit,
            actuation_threshold=p.actuation_threshold,
            task=task,
            target_range=target_range or (0.1 * p.travel, 0.9 * p.travel),
            max_episode_steps=max_episode_steps,
            n_substeps=n_substeps,
            render_mode=render_mode,
            geom_size=(0.01, p.lever_arm, 0.0),
        )

