"""Scissors reach task with bounded setpoint slew and palm-sphere grasp.

The original ``simple`` task is preserved for old models. This is a new task
version because it changes both the transition dynamics and success semantics.
"""
from dataclasses import asdict, dataclass
import re

import gymnasium as gym
import mujoco
import numpy as np

from .scissors_reach import ScissorsReachWrapper


@dataclass(frozen=True)
class SmoothReachConfig:
    # Paper15 uses 0.05 per 0.04 s here; Push/Spray use 0.025 per 0.02 s
    # through ToolActionSlewWrapper, matching 1.25 action units/s.
    action_delta_fraction: float = 0.05
    request_gap_penalty_weight: float = 1.0


class ScissorsSmoothReachWrapper(ScissorsReachWrapper):
    def __init__(self, env, *, smooth_config=None, **kwargs):
        config = smooth_config or SmoothReachConfig()
        if (not np.isfinite(config.action_delta_fraction)
                or not 0 < config.action_delta_fraction <= 1
                or not np.isfinite(config.request_gap_penalty_weight)
                or config.request_gap_penalty_weight < 0):
            raise ValueError('invalid smooth control configuration')
        self.smooth_config = config
        super().__init__(env, **kwargs)
        model = self.unwrapped.model
        self._mirror_geoms = set()
        self._moving_geoms = set()
        ordinary = {}
        mirrored = {}
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
            match = re.match(r'scissors_part(\d+)', name)
            if match:
                part = int(match[1])
                (mirrored if '_mirror' in name else ordinary).setdefault(part, set()).add(gid)
        if any(part >= 29 for part in ordinary):
            self._moving_geoms = set().union(*(geoms for part, geoms in ordinary.items() if part >= 29))
            self._mirror_geoms = mirrored.get(1, set())
        elif ordinary and mirrored:
            # Compact Scissors 1/2 assets have no part29..36 handle chain.
            self._moving_geoms = ordinary[max(ordinary)]
            self._mirror_geoms = mirrored[max(mirrored)]
        if not self._mirror_geoms or not self._moving_geoms:
            raise ValueError('both scissors handles need collision geometries')
        # The parent reads this set when constructing the contact observation.
        self._handles = self._mirror_geoms | self._moving_geoms
        self.applied_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._requested_gap_cost = 0.0

    def _handle_fingers(self):
        moving, mirror = set(), set()
        for contact in self.unwrapped.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            for tool, hand in ((a, b), (b, a)):
                if hand not in self._digits:
                    continue
                if tool in self._moving_geoms:
                    moving.add(self._digits[hand])
                if tool in self._mirror_geoms:
                    mirror.add(self._digits[hand])
        return moving, mirror

    def _success_info(self, info, pos, rotation):
        result = super()._success_info(info, pos, rotation)
        moving, mirror = self._handle_fingers()
        supported = bool(moving and mirror and len(moving | mirror) >= self.required_grasp_contacts
                         and not result['task_failure'])
        # Contacts are observable diagnostics and optional reward shaping only;
        # they must never override the palm-sphere grasp or RL success rule.
        result.update(task_version='scissors_reach_smooth_palm',
            grasp_moving_finger_count=len(moving), grasp_mirror_finger_count=len(mirror),
            grasp_supports_both_handles=supported,
            grasp_contact_fraction=(int(bool(moving)) + int(bool(mirror))) / 2.0
                * min(len(moving | mirror) / self.required_grasp_contacts, 1.),
            precision_success=float(result['is_on_palm'] and float(info['angle_error']) <= .01))
        return result

    def _augment_info(self, info):
        result = super()._augment_info(info)
        result['requested_applied_gap_cost'] = self._requested_gap_cost
        return result

    def reset(self, **kwargs):
        self.applied_action[:] = 0.
        self._requested_gap_cost = 0.
        obs, info = super().reset(**kwargs)
        info['applied_action'] = self.applied_action.copy()
        return obs, info

    def step(self, action):
        requested = np.asarray(action, dtype=np.float32)
        if requested.shape != self.action_space.shape:
            raise ValueError('wrong action shape')
        if not np.isfinite(requested).all():
            return super().step(requested)
        requested = np.clip(requested, -1., 1.)
        previous = self.applied_action.copy()
        limit = float(self.smooth_config.action_delta_fraction)
        applied = np.clip(previous + np.clip(requested - previous, -limit, limit), -1., 1.).astype(np.float32)
        self._requested_gap_cost = float(np.mean(np.square(requested - applied)))
        obs, reward, terminated, truncated, info = super().step(applied)
        if not info.get('numerical_failure', False):
            self.applied_action = applied
        info.update(applied_action=self.applied_action.copy(),
            requested_action=requested.copy(),
            requested_applied_gap_cost=self._requested_gap_cost,
            action_rate_limited_fraction=float(np.mean(np.abs(requested-applied) > 1e-6)),
            applied_action_delta_max=float(np.max(np.abs(self.applied_action - previous))))
        return obs, reward, terminated, truncated, info

    def _reward_terms(self, achieved_goal, desired_goal, info):
        terms = super()._reward_terms(achieved_goal, desired_goal, info)
        size = len(np.asarray(achieved_goal).reshape(-1))
        infos = [info]*size if isinstance(info, dict) else list(info)
        cost = np.asarray([item.get('requested_applied_gap_cost', 0.) for item in infos], dtype=np.float32)
        failed = np.asarray([item.get('task_failure', False) for item in infos], dtype=bool)
        penalty = np.where(failed, 0., -self.smooth_config.request_gap_penalty_weight * cost).astype(np.float32)
        terms['request_gap_penalty'] = penalty
        terms['total'] = terms['total'] + penalty
        return terms

    def goal_sampling_report(self):
        return {**super().goal_sampling_report(), 'task_version': 'scissors_reach_smooth_palm',
                'contact_unit': 'distinct_fingers_on_both_handles_diagnostic_only',
                'smooth_config': asdict(self.smooth_config)}
