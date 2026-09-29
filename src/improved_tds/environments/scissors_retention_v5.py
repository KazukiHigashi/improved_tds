"""Shared v5 acquisition/retention training; unchanged strict 2 s success test."""
from collections import deque
import gymnasium as gym
import mujoco
import numpy as np
from .scissors_control import ScissorsControlWrapper


def warning_counts(raw):
    return np.array([raw.data.warning[int(k)].number for k in
        (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
         mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL)])


class ScissorsRetentionV5Wrapper(ScissorsControlWrapper):
    durations = (.4, .8, 1.2, 2.)

    def __init__(self, env, *, retention_training_steps=0, acquisition_seconds=.4, **kwargs):
        self.training_curriculum = retention_training_steps > 0
        self.acquisition_seconds = float(acquisition_seconds)
        if not np.isfinite(self.acquisition_seconds) or self.acquisition_seconds < .12:
            raise ValueError('acquisition window must be finite and >= .12 s')
        self.stage = 0
        self.competence = deque(maxlen=20)
        self.retention_steps = 0
        self.numerical_failures = 0
        self._reset_phase()
        super().__init__(env, **kwargs)
        size = self.observation_space['observation'].shape[0] + 9
        self.observation_space = gym.spaces.Dict({**self.observation_space.spaces,
            'observation': gym.spaces.Box(-np.inf, np.inf, (size,), dtype=np.float32)})

    def _reset_phase(self):
        self.episode_step = 0
        self.acquired = False
        self.freeze_remaining = self.freeze_good = self.ready = self.cooldown = 0
        self.certified = False
        self.episode_seconds = 2.

    def _augment_info(self, info):
        result = super()._augment_info(info)
        self.acquired |= bool(result['grasp_maintained'])
        # Only initial contact acquisition gets a longer window. Established
        # grasp loss retains the original .12 s grace; pose/drop limits never change.
        acquiring = not self.acquired and self.episode_step*self.unwrapped.dt < self.acquisition_seconds
        if acquiring:
            result['failure_contact'] = False
            result['task_failure'] = bool(result['failure_displacement'] or result['failure_rotation']
                                          or result['object_dropped'])
        result['acquiring_initial_grasp'] = acquiring
        result['support_fraction'] = min(1., result['grasp_contact_count']/self.criteria.min_support_bodies)
        velocity = self.unwrapped.data.cvel[self._root_id]
        result['root_angular_speed'] = float(np.linalg.norm(velocity[:3]))
        result['root_linear_speed'] = float(np.linalg.norm(velocity[3:]))
        return result

    def _obs(self, obs):
        final = int(np.ceil(2./self.unwrapped.dt-1e-10))
        extra = np.array([self.freeze_remaining/final, min(self.freeze_good/final,1.),
            min(self.ready/10.,1.), self.episode_seconds/2., float(self.acquired),
            max(0.,1-self.episode_step*self.unwrapped.dt/self.acquisition_seconds),
            min(1.,self.episode_step/400.), self.cooldown/10., float(self.certified)],dtype=np.float32)
        return {**obs,'observation':np.concatenate([obs['observation'],extra])}

    def reset(self, **kwargs):
        self._reset_phase()
        obs, info = super().reset(**kwargs)
        if self.training_curriculum:
            # Full-range final-duration exposure is never absent even early on.
            self.episode_seconds = 2. if self._rng.random()<.2 else self.durations[self.stage]
        self._safe_obs = {k:np.array(v,copy=True) for k,v in obs.items()}
        self._safe_info = dict(info)
        info.update(is_stable_success=0., success_hold_steps=int(np.ceil(2./self.unwrapped.dt-1e-10)),
                    task_version='scissors_retention_v5', retention_seconds=self.episode_seconds)
        if not all(np.isfinite(x).all() for x in obs.values()):
            raise ValueError('nonfinite initial pose; refuse to start training')
        return self._obs(obs),info

    def _record_probe(self, success):
        if self.training_curriculum and self.episode_seconds==self.durations[self.stage] and self.stage<3:
            self.competence.append(bool(success))
            if len(self.competence)==20 and np.mean(self.competence)>=.7:
                self.stage+=1
                self.competence.clear()

    def _advance_phase(self, info, command_change, done):
        """Phase transitions isolated for testing; no simulator state restoration."""
        frozen = self.freeze_remaining>0 or self.certified
        good = bool(info.get('is_success',False)) and not done
        strict = False
        milestone = False
        aborted = False
        if frozen:
            self.freeze_good = self.freeze_good+1 if good else 0
            if not self.certified:
                self.freeze_remaining -= 1
                if not good:
                    self._record_probe(False)
                    self.freeze_remaining = self.freeze_good = 0
                    self.ready = 0
                    self.cooldown = 10
                    aborted = True
                elif self.freeze_remaining==0:
                    self._record_probe(True)
                    milestone = True
                    if self.freeze_good>=int(np.ceil(2./self.unwrapped.dt-1e-10)):
                        self.certified = True
                    else:
                        self.ready = 0
                        self.cooldown = 10
            strict = self.certified and good and self.freeze_good>=int(np.ceil(2./self.unwrapped.dt-1e-10))
        else:
            self.cooldown=max(0,self.cooldown-1)
            settled=(good and command_change<=.02 and info.get('root_linear_speed',np.inf)<=.05
                     and info.get('root_angular_speed',np.inf)<=.5)
            self.ready = self.ready+1 if settled and self.cooldown==0 else 0
            if self.ready>=int(np.ceil(.4/self.unwrapped.dt-1e-10)):
                self.freeze_remaining=int(np.ceil(self.episode_seconds/self.unwrapped.dt-1e-10))
                self.freeze_good=self.ready=0
        bonus=(2. if frozen and good else 0.)+(5. if milestone else 0.)
        if done:bonus=0.
        return dict(is_stable_success=float(strict), retention_frozen=frozen,
            retention_curriculum_success=milestone, retention_aborted=aborted,
            retention_success_steps=self.freeze_good, reward_retention=bonus,
            retention_seconds=self.episode_seconds, retention_stage=self.stage,
            success_hold_steps=int(np.ceil(2./self.unwrapped.dt-1e-10)),
            post_success_fixed_command=self.certified)

    def _numerical_failure(self):
        self.numerical_failures+=1
        self._record_probe(False)
        info={**self._safe_info,'numerical_failure':True,'task_failure':True,
              'grasp_maintained':0.,'is_success':0.,'is_stable_success':0.,
              'reward_retention':0.,'reward_total':-self.failure_penalty,
              'retention_frozen':False,'retention_stage':self.stage,
              'task_version':'scissors_retention_v5'}
        # Safe observation is a terminal sentinel, NOT a restored physics state.
        # The replay buffer discards this entire vector transition, and the
        # trajectory collector discards this episode before reading bad physics.
        return self._obs(self._safe_obs),-float(self.failure_penalty),True,False,info

    def step(self, action):
        self.retention_steps+=1
        self.episode_step+=1
        frozen=self.freeze_remaining>0 or self.certified
        request=self.applied_action.copy() if frozen else np.asarray(action)
        if not np.isfinite(request).all():return self._numerical_failure()
        before_action=self.applied_action.copy()
        before_time=float(self.unwrapped.data.time)
        before_warning=warning_counts(self.unwrapped)
        obs,reward,done,truncated,info=super().step(request)
        raw=self.unwrapped
        bad=(float(raw.data.time)<before_time or np.any(warning_counts(raw)>before_warning)
             or not np.isfinite(raw.data.qpos).all() or not np.isfinite(raw.data.qvel).all()
             or not np.isfinite(reward) or not all(np.isfinite(x).all() for x in obs.values()))
        if bad:return self._numerical_failure()
        delta=float(np.sqrt(np.mean((self.applied_action-before_action)**2)))
        phase=self._advance_phase(info,delta,bool(done))
        info={**info,**phase,'numerical_failure':False,'task_version':'scissors_retention_v5',
              'applied_command_change':delta}
        reward=float(reward)+phase['reward_retention']
        info['reward_total']=reward
        self._safe_obs={k:np.array(v,copy=True) for k,v in obs.items()}
        self._safe_info=dict(info)
        # SUCCESS DOES NOT TERMINATE: keep exactly the same command and collect
        # supported-retention reward to the common horizon or physical failure.
        return self._obs(obs),reward,done,truncated,info

    def _reward_terms(self, achieved_goal, desired_goal, info):
        terms=super()._reward_terms(achieved_goal,desired_goal,info)
        n=len(np.asarray(achieved_goal).reshape(-1))
        infos=[info]*n if isinstance(info,dict) else list(info)
        failed=np.array([x.get('task_failure',False) for x in infos])
        contact=np.array([x.get('support_fraction',0.) for x in infos])
        speed=np.array([x.get('root_linear_speed',0.) for x in infos])
        angular=np.array([x.get('root_angular_speed',0.) for x in infos])
        support=np.where(failed,0.,.5*contact).astype(np.float32)
        braking=np.where(failed,0.,-.25*np.minimum((speed/.05)**2+(angular/.5)**2,1.)).astype(np.float32)
        terms.update(support_acquisition=support,root_motion_penalty=braking)
        terms['total']=terms['total']+support+braking
        return terms

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward=super().compute_reward(achieved_goal,desired_goal,info)
        infos=[info] if isinstance(info,dict) else list(info)
        return reward+np.array([x.get('reward_retention',0.) for x in infos],dtype=np.float32)

    def goal_sampling_report(self):
        return {**super().goal_sampling_report(),'task_version':'scissors_retention_v5',
                'final_fixed_hold_seconds':2.,'training_retention_seconds':self.episode_seconds,
                'retention_stage':self.stage,'recent_probe_count':len(self.competence),
                'recent_probe_success':float(np.mean(self.competence)) if self.competence else None,
                'full_duration_probability':.2 if self.training_curriculum else 1.,
                'acquisition_seconds':self.acquisition_seconds,'success_terminates':False,
                'numerical_failures':self.numerical_failures}
