"""Versioned Scissors posture-generation task; identical criteria for variants 1--5."""
from dataclasses import asdict, dataclass
import re

import mujoco
import gymnasium as gym
import numpy as np

from .pose_pool import ScissorsGoalWrapper


@dataclass(frozen=True)
class PostureCriteria:
    angle_tolerance: float = 0.03
    hold_seconds: float = 0.4
    max_tool_speed: float = 0.10
    min_support_bodies: int = 2
    max_displacement: float = 0.10
    max_rotation: float = 0.8
    drop_distance: float = 0.08
    contact_grace_seconds: float = 0.12
    full_range_probability: float = 0.5

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for k,v in asdict(self).items() if k != 'full_range_probability'):
            raise ValueError('physical posture criteria must be finite and positive')
        if not 0 <= self.full_range_probability <= 1:
            raise ValueError('full-range probability must lie in [0,1]')


class ScissorsPostureWrapper(ScissorsGoalWrapper):
    """Contact-supported low-error posture, with explicit goal-distribution audit.

    Limits are provisional physical criteria, not claimed calibrated constants.
    Root-pose drift is relative to each authored reset, not the scene's fixed site.
    """

    def __init__(self, env, *, criteria=None, **kwargs):
        self.criteria = criteria or PostureCriteria()
        c = self.criteria
        kwargs['success_hold_steps'] = int(np.ceil(c.hold_seconds / env.unwrapped.dt))
        env.unwrapped.angle_threshold = c.angle_tolerance
        super().__init__(env, **kwargs)
        self.goal_histogram = np.zeros(10, dtype=int)
        self._full_bins = []
        self._lost_steps = 0
        self._start_pos = None
        self._start_rotation = None
        m = self.unwrapped.model
        self._root_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'scissors_part0')
        if self._root_id < 0:
            raise ValueError('missing scissors root body')
        self._handle_geoms = set()
        self._hand_geoms = set()
        for i in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ''
            match = re.match(r'scissors_part(\d+)', name)
            if match and int(match.group(1)) >= 2:
                self._handle_geoms.add(i)
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i])) or ''
            if body.startswith(('rh_', 'robot0:')):
                self._hand_geoms.add(i)
        if not self._handle_geoms or not self._hand_geoms:
            raise ValueError('handle/hand contact geometry mapping is empty')
        space = self.observation_space['observation']
        self.observation_space = gym.spaces.Dict({**self.observation_space.spaces,
            'observation': gym.spaces.Box(-np.inf,np.inf,(space.shape[0]+27,),dtype=np.float32)})

    def _state_observation(self, obs, info):
        u = self.unwrapped
        jid = mujoco.mj_name2id(u.model,mujoco.mjtObj.mjOBJ_JOINT,'scissors_hinge_2:joint')
        extra = np.concatenate([
            [u.data.qvel[u.model.jnt_dofadr[jid]]],
            (u.data.xpos[self._root_id]-self._start_pos)/self.criteria.max_displacement,
            (self._start_rotation.T@u.data.xmat[self._root_id].reshape(3,3)).ravel(),
            u.data.cvel[self._root_id],
            [info['grasp_contact_count']/self.criteria.min_support_bodies,
             self._lost_steps/max(1,np.ceil(self.criteria.contact_grace_seconds/u.dt))],
            [float(i in self._support_bodies) for i in self._observed_support_bodies],
        ]).astype(np.float32)
        return {**obs,'observation':np.concatenate([obs['observation'],extra])}

    def set_target_max_fraction(self, maximum):
        if not self.target_min_fraction < maximum <= 1:
            raise ValueError('invalid curriculum maximum')
        self.target_max_fraction = float(maximum)
        return self.target_max_fraction

    def goal_sampling_report(self):
        return dict(task_version='scissors_posture_v2', criteria=asdict(self.criteria),
                    effective_curriculum_max=self.target_max_fraction,
                    sampled_goal_bins=self.goal_histogram.tolist())

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._full_bins = []
        options = dict(options or {})
        if 'target_angle' not in options:
            if self._rng.random() < self.criteria.full_range_probability:
                if not self._full_bins:
                    self._full_bins = self._rng.permutation(10).tolist()
                fraction = (self._full_bins.pop() + self._rng.random()) / 10
            else:
                fraction = self._rng.uniform(self.target_min_fraction, self.target_max_fraction)
            low, high = self.unwrapped.target_angle_range
            options['target_angle'] = float(low + fraction * (high - low))
            self.goal_histogram[min(9, int(fraction * 10))] += 1
        self._start_pos = None
        self._start_rotation = None
        self._lost_steps = 0
        obs, info = super().reset(seed=seed, options=options)
        self._success_streak = 0
        info['success_streak'] = 0
        info['is_stable_success'] = 0.0
        return self._state_observation(obs,info), info

    def _augment_info(self, info):
        result = dict(info)
        u, c = self.unwrapped, self.criteria
        pos = u.data.xpos[self._root_id].copy()
        rotation = u.data.xmat[self._root_id].reshape(3, 3).copy()
        if self._start_pos is None:
            self._start_pos, self._start_rotation = pos.copy(), rotation.copy()
        displacement = float(np.linalg.norm(pos - self._start_pos))
        rotation_error = float(np.arccos(np.clip((np.trace(self._start_rotation.T @ rotation)-1)/2,-1,1)))
        support = set()
        for contact in u.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if a in self._handle_geoms and b in self._hand_geoms:
                support.add(int(u.model.geom_bodyid[b]))
            if b in self._handle_geoms and a in self._hand_geoms:
                support.add(int(u.model.geom_bodyid[a]))
        self._support_bodies = support
        # Five digit support indicators, derived from ancestry rather than names
        # of individual distal links. A sixth slot records palm support.
        groups = [set() for _ in range(6)]
        for bid in support:
            name = mujoco.mj_id2name(u.model,mujoco.mjtObj.mjOBJ_BODY,bid) or ''
            assigned=False
            for k,prefix in enumerate(('ff','mf','rf','lf','th')):
                if prefix in name.lower():
                    groups[k].add(bid); assigned=True
            if not assigned: groups[5].add(bid)
        self._observed_support_bodies = [next(iter(g),-1) for g in groups]
        jid = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_JOINT, 'scissors_hinge_2:joint')
        speed = abs(float(u.data.qvel[u.model.jnt_dofadr[jid]]))
        dropped = bool(pos[2] < self._start_pos[2] - c.drop_distance)
        supported = len(support) >= c.min_support_bodies
        self._lost_steps = 0 if supported else self._lost_steps + 1
        pose_ok = displacement <= c.max_displacement and rotation_error <= c.max_rotation and not dropped
        grasped = supported and pose_ok
        failure = (not pose_ok) or self._lost_steps > int(np.ceil(c.contact_grace_seconds / u.dt))
        error = float(result['angle_error'])
        success = grasped and error <= c.angle_tolerance and speed <= c.max_tool_speed
        self._success_streak = self._success_streak + 1 if success else 0
        result.update(grasp_maintained=float(grasped), is_success=float(success),
                      is_stable_success=float(self._success_streak >= self.success_hold_steps),
                      success_streak=self._success_streak, success_hold_steps=self.success_hold_steps,
                      is_in_grasp_space=float(grasped), grasp_contact_count=len(support),
                      tool_speed=speed, object_displacement=displacement, object_rotation_error=rotation_error,
                      object_dropped=dropped, task_failure=failure,
                      failure_contact=bool(self._lost_steps > int(np.ceil(c.contact_grace_seconds / u.dt))),
                      failure_displacement=bool(displacement > c.max_displacement),
                      failure_rotation=bool(rotation_error > c.max_rotation),
                      initial_achieved_goal=self._initial_achieved,
                      precision_success=float(grasped and error <= 0.01 and speed <= c.max_tool_speed))
        return result

    def step(self, action):
        # Parent uses our support test but immediate loss termination is replaced
        # with the explicit contact grace (raw environment never terminates).
        obs, reward, _, truncated, info = super().step(action)
        return self._state_observation(obs,info), reward, bool(info['task_failure']), truncated, info

    def _reward_terms(self, achieved_goal, desired_goal, info):
        terms = super()._reward_terms(achieved_goal, desired_goal, info)
        achieved = np.asarray(achieved_goal).reshape(-1)
        desired = np.asarray(desired_goal).reshape(-1)
        infos = [info] * len(achieved) if isinstance(info, dict) else list(info)
        failure = np.array([i.get('task_failure', False) for i in infos])
        grasp = np.array([i.get('grasp_maintained', 0) for i in infos]) > .5
        slow = np.array([i.get('tool_speed', np.inf) for i in infos]) <= self.criteria.max_tool_speed
        success = grasp & slow & (abs(achieved - desired) <= self.criteria.angle_tolerance)
        terms['success_bonus'] = (self.success_bonus * success).astype(np.float32)
        terms['failure_penalty'] = (-self.failure_penalty * failure).astype(np.float32)
        names = ['tracking','progress','grasp','interaction','success_bonus','control_delta_penalty','control_magnitude_penalty']
        terms['total'] = np.where(failure, terms['failure_penalty'], sum(terms[n] for n in names)).astype(np.float32)
        return terms
