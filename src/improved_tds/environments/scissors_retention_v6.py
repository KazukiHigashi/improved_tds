"""Reach then hold: accessible probes, bounded settling, unchanged certification.

All Scissors variants share these rules. Training-only duration shaping is never
reported as task success; validation always requires 50 consecutive good frames
under one fixed actuator command (dt=.04 s).
"""
import numpy as np

from .scissors_retention_v5 import ScissorsRetentionV5Wrapper


class ScissorsRetentionV6Wrapper(ScissorsRetentionV5Wrapper):
    readiness_seconds = .12
    settling_seconds = .4

    def __init__(self, env, *, retention_training_steps=0, **kwargs):
        self.training_budget = int(retention_training_steps)
        self.probe_attempts = self.probe_completions = self.probe_successes = 0
        self.instant_good_frames = 0
        super().__init__(env, retention_training_steps=retention_training_steps, **kwargs)

    def _promote_by_budget(self):
        if self.training_curriculum:
            # Competence can promote earlier; lack of completed probes must not
            # leave the entire 3M-step run at .4 s. Final 40% is exclusively 2 s.
            fraction = self.retention_steps / max(1, self.training_budget)
            stage = sum(fraction >= boundary for boundary in (.2, .4, .6))
            if stage > self.stage:
                self.stage = stage
                self.competence.clear()

    def reset(self, **kwargs):
        if getattr(self, 'freeze_remaining', 0) > 0 and not self.certified:
            self._record_probe(False)  # horizon reached before certification
        self._promote_by_budget()
        obs, info = super().reset(**kwargs)
        info['task_version'] = 'scissors_retention_v6'
        return obs, info

    def _record_probe(self, success):
        self.probe_completions += 1
        self.probe_successes += int(success)
        super()._record_probe(success)

    def _advance_phase(self, info, command_change, done):
        frozen = self.freeze_remaining > 0 or self.certified
        good = bool(info.get('is_success', False)) and not done
        self.instant_good_frames += int(good)
        final_steps = int(np.ceil(2. / self.unwrapped.dt - 1e-10))
        needed = int(np.ceil(self.episode_seconds / self.unwrapped.dt - 1e-10))
        milestone = aborted = strict = False
        if frozen:
            self.freeze_good = self.freeze_good + 1 if good else 0
            if not self.certified:
                self.freeze_remaining -= 1
                milestone = self.freeze_good >= needed and not done
                # Allow a bounded settling transient; the qualifying streak
                # still contains ONLY uninterrupted original-success frames.
                if milestone or done or self.freeze_remaining <= 0:
                    self._record_probe(milestone)
                    self.freeze_remaining = 0
                    self.ready = 0
                    if milestone and self.freeze_good >= final_steps:
                        self.certified = True
                    else:
                        aborted = not milestone
                        self.cooldown = 10
            strict = self.certified and good and self.freeze_good >= final_steps
        else:
            self.cooldown = max(0, self.cooldown - 1)
            # Applied command is forcibly held next step. It need not already
            # be motionless under stochastic SAC exploration to try holding.
            # Root motion remains penalized; all physical failure limits stay.
            self.ready = self.ready + 1 if good and self.cooldown == 0 else 0
            if self.ready >= int(np.ceil(self.readiness_seconds / self.unwrapped.dt - 1e-10)):
                self.freeze_remaining = needed + int(np.ceil(self.settling_seconds / self.unwrapped.dt - 1e-10))
                self.freeze_good = self.ready = 0
                self.probe_attempts += 1
        # Duration-scaled milestone prevents cheap short probes from receiving
        # the same completion bonus as the final 2-second task.
        # Move 80% of the existing instantaneous-success bonus to fixed-command
        # success. Feedback tracking alone must not be almost as rewarding as
        # learning a passively stable posture. All variants share this split.
        bonus = ((2. + .8 * self.success_bonus) if frozen and good else 0.) + (5. * self.episode_seconds / 2. if milestone else 0.)
        if done:
            bonus = 0.
        return dict(is_stable_success=float(strict), retention_frozen=frozen,
            retention_curriculum_success=milestone, retention_aborted=aborted,
            retention_success_steps=self.freeze_good, reward_retention=bonus,
            retention_seconds=self.episode_seconds, retention_stage=self.stage,
            success_hold_steps=final_steps, post_success_fixed_command=self.certified,
            retention_probe_attempts=self.probe_attempts,
            retention_probe_completions=self.probe_completions,
            retention_probe_successes=self.probe_successes)

    def _numerical_failure(self):
        # Numeric failure outside a hold is NOT a failed retention probe.
        active_probe = self.freeze_remaining > 0 and not self.certified
        if active_probe:
            self._record_probe(False)
        self.freeze_remaining = 0
        self.numerical_failures += 1
        info = {**self._safe_info, 'numerical_failure': True, 'task_failure': True,
            'grasp_maintained': 0., 'is_success': 0., 'is_stable_success': 0.,
            'reward_retention': 0., 'reward_total': -self.failure_penalty,
            'retention_frozen': active_probe, 'retention_aborted': active_probe,
            'retention_stage': self.stage, 'task_version': 'scissors_retention_v6'}
        return self._obs(self._safe_obs), -float(self.failure_penalty), True, False, info

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        info['task_version'] = 'scissors_retention_v6'
        return obs, reward, done, truncated, info

    def _reward_terms(self, achieved_goal, desired_goal, info):
        terms = super()._reward_terms(achieved_goal, desired_goal, info)
        count = len(np.asarray(achieved_goal).reshape(-1))
        infos = [info] * count if isinstance(info, dict) else list(info)
        failed = np.array([i.get('task_failure', False) for i in infos])
        old_bonus = terms['success_bonus'].copy()
        terms['success_bonus'] = .2 * old_bonus
        terms['total'] = terms['total'] - np.where(failed, 0., .8 * old_bonus)
        return terms

    def goal_sampling_report(self):
        return {**super().goal_sampling_report(), 'task_version': 'scissors_retention_v6',
            'readiness_seconds': self.readiness_seconds, 'settling_budget_seconds': self.settling_seconds,
            'hard_command_change_gate': False, 'hard_root_speed_gate': False,
            'feedback_success_bonus_fraction': .2, 'fixed_success_bonus_fraction': .8,
            'budget_stage_boundaries': [.2, .4, .6],
            'probe_attempts': self.probe_attempts, 'probe_completions': self.probe_completions,
            'probe_successes': self.probe_successes, 'instant_good_frames': self.instant_good_frames,
            'full_duration_probability': 1. if not self.training_curriculum or self.stage == 3 else .2}
