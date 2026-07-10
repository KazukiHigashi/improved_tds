from __future__ import annotations

from dataclasses import dataclass

from improved_tds.environments.one_dof import OneDoFToolEnv


@dataclass(frozen=True)
class ButtonParameters:
    travel: float = 0.01
    stiffness: float = 30.0
    damping: float = 0.2
    actuation_threshold: float = 0.007
    radius: float = 0.012
    body_friction: float = 0.5
    neighboring_buttons: int = 0
    body_mass: float = 0.02
    force_limit: float = 5.0


class ButtonEnv(OneDoFToolEnv):
    def __init__(
        self,
        parameters: ButtonParameters | None = None,
        *,
        instance_id: str = "button-default",
        task: str = "single_button_press",
        target_range: tuple[float, float] | None = None,
        max_episode_steps: int = 200,
        n_substeps: int = 10,
        render_mode: str | None = None,
    ):
        self.parameters = parameters or ButtonParameters()
        p = self.parameters
        super().__init__(
            family="button",
            instance_id=instance_id,
            joint_type="slide",
            travel=p.travel,
            stiffness=p.stiffness,
            damping=p.damping,
            friction=0.0,
            preload=0.0,
            mass=p.body_mass,
            force_limit=p.force_limit,
            actuation_threshold=p.actuation_threshold,
            task=task,
            target_range=target_range or (0.5 * p.travel, 0.9 * p.travel),
            max_episode_steps=max_episode_steps,
            n_substeps=n_substeps,
            render_mode=render_mode,
            geom_size=(p.radius, 0.005, 0.0),
        )

