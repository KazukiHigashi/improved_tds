"""Reach a tool angle with the hinge inside the original palm-attached sphere.

No command freeze, speed criterion, retention curriculum or contact gate.
Contacts and physical motion remain diagnostics; only numerical failure is
terminal before the usual episode time limit.
"""
import gymnasium as gym
import mujoco
import numpy as np
import re
from types import MethodType
from .pose_pool import ScissorsGoalWrapper
from .scissors_retention_v5 import warning_counts


def _palm_attached_grasp_center(raw, radius=None):
    """Original green-sphere centre, rigidly attached to the palm frame."""
    del radius
    palm=raw._tds_palm_body_id
    rotation=raw.data.xmat[palm].reshape(3,3)
    return raw.data.xpos[palm]+rotation@raw._tds_palm_sphere_local_center


class ScissorsReachWrapper(ScissorsGoalWrapper):
    def __init__(self, env, *, required_grasp_contacts=2, angle_tolerance=.03,
                 contact_auxiliary_reward_weight=0., **kwargs):
        super().__init__(env, **kwargs)
        self.unwrapped.angle_threshold = float(angle_tolerance)
        if not np.isfinite(angle_tolerance) or angle_tolerance <= 0: raise ValueError('positive angle tolerance required')
        self.required_grasp_contacts = int(required_grasp_contacts)
        if not 1 <= self.required_grasp_contacts <= 5:
            raise ValueError('required grasp contacts must be 1..5 fingers')
        if not np.isfinite(contact_auxiliary_reward_weight) or contact_auxiliary_reward_weight<0:
            raise ValueError('contact auxiliary reward weight must be finite and non-negative')
        self.contact_auxiliary_reward_weight=float(contact_auxiliary_reward_weight)
        m = self.unwrapped.model
        raw=self.unwrapped
        palm=next((mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
                   for name in ('rh_palm','robot0:palm')
                   if mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)>=0),-1)
        if palm<0:
            raise ValueError('palm body not found for palm-attached grasp sphere')
        # Capture the original XML initial configuration BEFORE the pose-pool
        # wrapper changes the hand mount. This exactly preserves the green
        # sphere's initial offset and radius in the palm coordinate frame.
        original_center=np.asarray(raw._grasp_center_space(radius=raw.grasp_radius),dtype=float)
        raw._tds_palm_body_id=palm
        raw._tds_palm_sphere_local_center=(raw.data.xmat[palm].reshape(3,3).T@
                                            (original_center-raw.data.xpos[palm])).copy()
        raw._grasp_center_space=MethodType(_palm_attached_grasp_center,raw)
        self._root_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'scissors_part0')
        hinge = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'scissors_hinge_2:joint')
        self._hinge_dof = int(m.jnt_dofadr[hinge])
        self._handles = set()
        self._digits = {}
        for gid in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
            if name.startswith('scissors_part'):
                match = re.match(r'scissors_part(\d+)', name)
                if match and int(match[1]) >= 2: self._handles.add(gid)
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gid])) or ''
            for i, digit in enumerate(('ff','mf','rf','lf','th')):
                if body.startswith(('rh_'+digit, 'robot0:'+digit)):
                    self._digits[gid] = i
        if self._root_id < 0 or not self._handles or not self._digits:
            raise ValueError('missing scissors contact geometry')
        self._start_pos = self._start_rotation = None
        self._contacts = np.zeros(5, dtype=np.float32)
        self.goal_bins = np.zeros(10, dtype=int)
        self.numerical_failures = 0
        size = self.observation_space['observation'].shape[0] + 17
        self.observation_space = gym.spaces.Dict({**self.observation_space.spaces,
            'observation': gym.spaces.Box(-np.inf, np.inf, (size,), dtype=np.float32)})

    def _obs(self, obs):
        raw = self.unwrapped
        extra = np.concatenate([(raw.data.xpos[self._root_id]-self._start_pos)/.1,
            (self._start_rotation.T@raw.data.xmat[self._root_id].reshape(3,3)).ravel(), self._contacts])
        return {**obs, 'observation': np.concatenate([obs['observation'],extra]).astype(np.float32)}

    def _augment_info(self, info):
        raw = self.unwrapped
        pos = raw.data.xpos[self._root_id].copy()
        rotation = raw.data.xmat[self._root_id].reshape(3,3).copy()
        if self._start_pos is None:
            self._start_pos, self._start_rotation = pos.copy(), rotation.copy()
        self._contacts[:] = 0
        for c in raw.data.contact:
            a,b = int(c.geom1),int(c.geom2)
            if a in self._handles and b in self._digits: self._contacts[self._digits[b]] = 1
            if b in self._handles and a in self._digits: self._contacts[self._digits[a]] = 1
        return self._success_info(info, pos, rotation)

    def _success_info(self, info, pos, rotation):
        displacement = float(np.linalg.norm(pos-self._start_pos))
        rotation_error = float(np.arccos(np.clip((np.trace(self._start_rotation.T@rotation)-1)/2,-1,1)))
        dropped = bool(pos[2] < self._start_pos[2]-.08)
        tipped = bool(rotation_error > .8)
        # Physical motion remains diagnostic. The grasp and RL success rules
        # have no contact, displacement, rotation, or drop gate.
        count = int(self._contacts.sum())
        sphere_center=self.unwrapped._grasp_center_space()
        hinge_center=self.unwrapped._scissors_joint_pos()
        palm_distance=float(np.linalg.norm(hinge_center-sphere_center))
        grasped=bool(palm_distance<self.unwrapped.grasp_threshold)
        success = grasped and float(info['angle_error']) <= self.unwrapped.angle_threshold
        self._success_streak = self._success_streak+1 if success else 0
        return {**info, 'task_version':'scissors_reach_palm', 'grasp_contact_count':count,
            'grasp_contact_fraction':min(count/self.required_grasp_contacts,1.),
            'is_on_palm':float(grasped),'palm_sphere_distance':palm_distance,
            'palm_sphere_radius':float(self.unwrapped.grasp_threshold),
            'palm_sphere_center':sphere_center.copy(),
            'is_in_grasp_space':float(grasped), 'grasp_maintained':float(grasped),
            'is_success':float(success),'is_stable_success':float(self._success_streak>=self.success_hold_steps),
            'success_streak':self._success_streak,'success_hold_steps':self.success_hold_steps,
            'object_dropped':dropped,'object_tipped':tipped,'task_failure':False,
            'failure_contact':False,'failure_rotation':tipped,'failure_displacement':displacement>.1,
            'object_displacement':displacement,'object_rotation_error':rotation_error,
            'initial_achieved_goal':self._initial_achieved,'numerical_failure':False,
            'tool_speed':abs(float(self.unwrapped.data.qvel[self._hinge_dof])),
            'speed_success_criterion':False,
            'angle_tolerance':float(self.unwrapped.angle_threshold),
            'precision_success':float(grasped and float(info['angle_error'])<=.01)}

    def reset(self, **kwargs):
        self._start_pos = self._start_rotation = None
        obs,info = super().reset(**kwargs)
        self._success_streak = 0
        info.update(success_streak=0,is_stable_success=0.)
        low,high = self.unwrapped.target_angle_range
        b = int(np.clip((float(obs['desired_goal'][0])-low)/(high-low)*10,0,9))
        self.goal_bins[b] += 1
        obs = self._obs(obs)
        if not all(np.isfinite(x).all() for x in obs.values()): raise ValueError('invalid reset state')
        self._safe_obs = {k:v.copy() for k,v in obs.items()}
        self._safe_info = dict(info)
        return obs,info

    def step(self, action):
        raw = self.unwrapped
        before_time, before_warning = float(raw.data.time), warning_counts(raw)
        if not np.isfinite(action).all(): return self._numerical_failure()
        obs,reward,_,truncated,info = super().step(action)
        if (raw.data.time < before_time or np.any(warning_counts(raw)>before_warning)
            or not np.isfinite(raw.data.qpos).all() or not np.isfinite(raw.data.qvel).all()
            or not np.isfinite(reward) or not all(np.isfinite(x).all() for x in obs.values())):
            return self._numerical_failure()
        obs = self._obs(obs)
        if not all(np.isfinite(x).all() for x in obs.values()): return self._numerical_failure()
        self._safe_obs = {k:v.copy() for k,v in obs.items()}
        self._safe_info = dict(info)
        # As in Push/Spray, contact shortage alone is not a terminal event.
        return obs,reward,bool(info['task_failure']),truncated,info

    def _numerical_failure(self):
        self.numerical_failures += 1
        return ({k:v.copy() for k,v in self._safe_obs.items()}, -self.failure_penalty,True,False,
            {**self._safe_info,'numerical_failure':True,'task_failure':True,
             'is_success':0.,'is_stable_success':0.,'reward_total':-self.failure_penalty})

    def _reward_terms(self, achieved_goal, desired_goal, info):
        n = len(np.asarray(achieved_goal).reshape(-1))
        infos = [info]*n if isinstance(info,dict) else list(info)
        terms = super()._reward_terms(achieved_goal, desired_goal, infos)
        failed = np.array([i.get('task_failure',False) for i in infos],dtype=bool)
        grasped = np.array([i.get('is_on_palm',False) for i in infos],dtype=bool)
        contact_fraction=np.array([i.get('grasp_contact_fraction',0.) for i in infos],dtype=np.float32)
        error = np.abs(np.asarray(achieved_goal).reshape(-1)-np.asarray(desired_goal).reshape(-1))
        terms['success_bonus'] = (self.success_bonus*(grasped & (error<=self.unwrapped.angle_threshold))).astype(np.float32)
        terms['contact_auxiliary']=(self.contact_auxiliary_reward_weight*contact_fraction).astype(np.float32)
        terms['failure_penalty'] = (-self.failure_penalty*failed).astype(np.float32)
        names = ('tracking','progress','grasp','interaction','success_bonus','contact_auxiliary',
                 'control_delta_penalty','control_magnitude_penalty')
        terms['total'] = np.where(failed,terms['failure_penalty'],sum(terms[k] for k in names)).astype(np.float32)
        return terms

    def goal_sampling_report(self):
        return dict(task_version='scissors_reach_palm',success_hold_steps=self.success_hold_steps,
            angle_tolerance=self.unwrapped.angle_threshold,required_grasp_contacts=self.required_grasp_contacts,
            grasp_criterion='hinge_center_inside_palm_attached_original_sphere',
            palm_sphere_radius=self.unwrapped.grasp_threshold,
            palm_sphere_local_center=self.unwrapped._tds_palm_sphere_local_center.tolist(),
            contact_auxiliary_reward_weight=self.contact_auxiliary_reward_weight,
            contact_unit='distinct_fingers_diagnostic_only',speed_success_criterion=False,fixed_command=False,
            sampled_goal_bins=self.goal_bins.tolist(),effective_curriculum_max=self.target_max_fraction,
            numerical_failures=self.numerical_failures)
