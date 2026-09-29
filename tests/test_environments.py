from __future__ import annotations

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

import improved_tds
from improved_tds.environments.pose_pool import (
    PushGoalWrapper,
    ScissorsGoalWrapper,
    SprayGoalWrapper,
    pool_filename_for_environment,
)


@pytest.mark.simulation
def test_scissors_goal_wrapper_samples_requested_full_range_band() -> None:
    pytest.importorskip("gym_scissors")
    from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments

    _register_tool_manip_gym_environments()
    wrapped = ScissorsGoalWrapper(
        gym.make("ScissorsVariant1-v0", hold_initial_steps=0),
        target_min_fraction=0.8,
        target_max_fraction=1.0,
        seed=7,
    )
    try:
        observation, _ = wrapped.reset(seed=7)
        assert 0.8 <= float(observation["desired_goal"][0]) <= 1.0
        observation, _ = wrapped.reset(seed=7, options={"target_angle": 0.25})
        assert float(observation["desired_goal"][0]) == pytest.approx(0.25)
    finally:
        wrapped.close()


@pytest.mark.simulation
@pytest.mark.parametrize(
    "env_id",
    [
        "ExpScissor1-v0",
        "ExpScissor2-v0",
        "ExpScissor3-v0",
        "ExpScissor4-v0",
        "ExpScissor5-v0",
        "TDS-Trigger-v0",
        "TDS-Button-v0",
    ],
)
def test_environment_reset_step_and_articulated_api(env_id) -> None:
    improved_tds.register_environments()
    env = gym.make(env_id)
    obs, _ = env.reset(seed=5)
    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    result = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert len(result) == 5
    unwrapped = env.unwrapped
    assert unwrapped.get_tool_state().shape == (1,)
    assert unwrapped.get_tool_state_rate().shape == (1,)
    assert "joint_torque" in unwrapped.get_force_observation()
    check_env(unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.simulation
def test_scissors_returned_observation_matches_internal_state() -> None:
    improved_tds.register_environments()
    env = gym.make("ExpScissor1-v0")
    env.reset(seed=8)
    returned, _, _, _, _ = env.step(
        np.full(env.action_space.shape, 0.7, dtype=np.float32)
    )
    internal = env.unwrapped._get_obs()
    for key in returned:
        np.testing.assert_allclose(returned[key], internal[key], atol=1e-7)
    env.close()


@pytest.mark.simulation
def test_scissors_stable_success_is_observable_but_not_her_terminal() -> None:
    improved_tds.register_environments()
    env = gym.make(
        "ExpScissor1-v0",
        angle_threshold=2.0,
        success_hold_steps=3,
        max_episode_steps=4,
    )
    env.reset(seed=9, options={"target": 0.0})
    infos = []
    results = []
    for _ in range(4):
        result = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        results.append(result)
        infos.append(result[-1])
    assert [info["success_streak"] for info in infos] == [1, 2, 3, 4]
    assert [info["is_stable_success"] for info in infos] == [0.0, 0.0, 1.0, 1.0]
    assert all(not result[2] for result in results)
    assert results[-1][3]
    env.close()


@pytest.mark.simulation
def test_scissors_reset_restores_complete_dynamic_state() -> None:
    improved_tds.register_environments()
    env = gym.make("ExpScissor1-v0", hold_initial_steps=2)
    env.reset(seed=11, options={"target": 0.3})
    initial = env.unwrapped.data
    expected = {
        "qpos": initial.qpos.copy(),
        "qvel": initial.qvel.copy(),
        "ctrl": initial.ctrl.copy(),
        "time": float(initial.time),
    }
    for _ in range(3):
        env.step(np.full(env.action_space.shape, 0.8, dtype=np.float32))
    env.reset(seed=11, options={"target": 0.3})
    np.testing.assert_allclose(env.unwrapped.data.qpos, expected["qpos"], atol=1e-10)
    np.testing.assert_allclose(env.unwrapped.data.qvel, expected["qvel"], atol=1e-10)
    np.testing.assert_allclose(env.unwrapped.data.ctrl, expected["ctrl"], atol=1e-10)
    np.testing.assert_allclose(env.unwrapped.data.qvel, 0.0, atol=1e-12)
    assert float(env.unwrapped.data.time) == pytest.approx(expected["time"])
    env.close()


@pytest.mark.simulation
def test_scissors_her_reward_preserves_grasp_condition() -> None:
    improved_tds.register_environments()
    env = gym.make("ExpScissor1-v0")
    unwrapped = env.unwrapped
    achieved = np.array([[0.2]], dtype=np.float32)
    desired = np.array([[0.2]], dtype=np.float32)
    in_grasp = unwrapped.compute_reward(
        achieved, desired, [{"is_in_grasp_space": 1.0}]
    )
    out_of_grasp = unwrapped.compute_reward(
        achieved, desired, [{"is_in_grasp_space": 0.0}]
    )
    np.testing.assert_array_equal(in_grasp, np.array([0.0], dtype=np.float32))
    np.testing.assert_array_equal(out_of_grasp, np.array([-1.0], dtype=np.float32))
    env.close()


@pytest.mark.simulation
@pytest.mark.parametrize(
    "env_id",
    [
        "ShadowHandSprayBottle-v0",
        "ShadowHandLiteSprayBottle-v0",
        "SprayBottleManipulate-v0",
        "ShadowHandPushSpray-v0",
        "ShadowHandLitePushSpray-v0",
        "PushSprayManipulate-v0",
        "ShadowHandSpray1-v0",
        "ShadowHandLiteSpray1-v0",
        "SprayVariant1-v0",
        "ShadowHandSpray2-v0",
        "ShadowHandLiteSpray2-v0",
        "SprayVariant2-v0",
        "ShadowHandSpray3-v0",
        "ShadowHandLiteSpray3-v0",
        "SprayVariant3-v0",
        "ShadowHandSpray4-v0",
        "ShadowHandLiteSpray4-v0",
        "SprayVariant4-v0",
        "ShadowHandSpray5-v0",
        "ShadowHandLiteSpray5-v0",
        "SprayVariant5-v0",
    ],
)
def test_spray_goal_dense_her_reward_for_all_spray_types(env_id: str) -> None:
    pytest.importorskip("gym_spray")
    env = gym.make(env_id, bottle_xy_noise=0.0)
    wrapped = SprayGoalWrapper(
        env,
        target_min_fraction=0.2,
        target_max_fraction=0.8,
        required_grasp_contacts=2,
        success_hold_steps=3,
        seed=17,
    )
    try:
        _, info = wrapped.reset(seed=17)
        limit = float(env.unwrapped.trigger_limit)
        assert 0.2 * limit <= info["target_trigger_angle"] <= 0.8 * limit
        assert info["trigger_tracking_tolerance"] == pytest.approx(
            env.unwrapped.trigger_tracking_tolerance
        )
        desired = np.asarray([info["target_trigger_angle"]], dtype=np.float32)
        no_grasp = wrapped.compute_reward(
            desired,
            desired,
            {**info, "grasp_contact_count": 0},
        )
        grasped = wrapped.compute_reward(
            desired,
            desired,
            {**info, "grasp_contact_count": 2},
        )
        assert float(np.asarray(grasped).reshape(-1)[0]) > float(
            np.asarray(no_grasp).reshape(-1)[0]
        )
        fallen = wrapped.compute_reward(
            desired,
            desired,
            {**info, "grasp_contact_count": 2, "bottle_tipped": True},
        )
        assert float(np.asarray(fallen).reshape(-1)[0]) == -wrapped.failure_penalty

        no_progress = wrapped.compute_reward(
            np.asarray([0.0], dtype=np.float32),
            desired,
            {**info, "grasp_contact_count": 0},
        )
        half_progress = wrapped.compute_reward(
            0.5 * desired,
            desired,
            {**info, "grasp_contact_count": 0},
        )
        assert float(half_progress[0]) > float(no_progress[0])

        far = wrapped.compute_reward(
            desired,
            desired,
            {**info, "grasp_contact_count": 0, "index_to_trigger_distance": 0.3},
        )
        close = wrapped.compute_reward(
            desired,
            desired,
            {**info, "grasp_contact_count": 0, "index_to_trigger_distance": 0.0},
        )
        assert float(close[0]) > float(far[0])

        tolerance = float(env.unwrapped.trigger_tracking_tolerance)
        within_tolerance = wrapped._reward_terms(
            desired + 0.5 * tolerance,
            desired,
            [{**info, "grasp_contact_count": 2}],
        )
        assert float(within_tolerance["success_bonus"][0]) == 5.0

        success_info = {
            **info,
            "trigger_target_reached": True,
            "grasp_contact_count": 2,
            "bottle_dropped": False,
            "bottle_tipped": False,
        }
        stable_infos = [wrapped._augment_info(success_info) for _ in range(3)]
        assert [item["success_streak"] for item in stable_infos] == [1, 2, 3]
        assert [item["is_stable_success"] for item in stable_infos] == [0.0, 0.0, 1.0]

        achieved_batch = np.asarray([[0.0], [desired[0]]], dtype=np.float32)
        desired_batch = np.repeat(desired.reshape(1, 1), 2, axis=0)
        batch_reward = wrapped.compute_reward(
            achieved_batch,
            desired_batch,
            [info, {**info, "grasp_contact_count": 2}],
        )
        assert batch_reward.shape == (2,)
        assert np.all(np.isfinite(batch_reward))
        assert wrapped.failure_penalty == pytest.approx(25.0)

        _, step_reward, _, _, step_info = wrapped.step(
            np.zeros(wrapped.action_space.shape, dtype=np.float32)
        )
        assert step_reward == pytest.approx(step_info["reward_total"])
        for term in (
            "tracking",
            "trigger_progress",
            "grasp",
            "index_proximity",
            "success_bonus",
            "failure_penalty",
        ):
            assert f"reward_{term}" in step_info
    finally:
        wrapped.close()

@pytest.mark.simulation
def test_push_goal_randomization_and_dense_her_reward() -> None:
    pytest.importorskip("gym_push")
    env = gym.make("ShadowHandLitePush1-v0", object_xy_noise=0.0)
    assert pool_filename_for_environment(env) == "push_lite_1.npz"
    wrapped = PushGoalWrapper(
        env,
        target_min_fraction=0.2,
        target_max_fraction=0.8,
        required_grasp_contacts=2,
        success_hold_steps=3,
    )
    observation, info = wrapped.reset(seed=19)
    target = float(observation["desired_goal"][0])
    limit = float(env.unwrapped.push_limit)
    assert 0.2 * limit <= target <= 0.8 * limit
    assert info["target_push_depth"] == pytest.approx(target)

    achieved = np.array([[target]], dtype=np.float32)
    desired = np.array([[target]], dtype=np.float32)
    valid = {
        "push_limit": limit,
        "push_tracking_tolerance": env.unwrapped.push_tracking_tolerance,
        "grasp_contact_count": 2,
        "thumb_on_plunger": False,
        "object_dropped": False,
        "object_tipped": False,
    }
    no_grasp = {**valid, "grasp_contact_count": 1}
    dropped = {**valid, "object_dropped": True}
    np.testing.assert_allclose(
        wrapped.compute_reward(achieved, desired, [valid]),
        np.array([8.0], dtype=np.float32),
    )
    # One contact of the two required: tracking pays 2.0 * 0.625, the common
    # grasp term pays 1.0 * 0.5, and the success bonus is withheld.
    np.testing.assert_allclose(
        wrapped.compute_reward(achieved, desired, [no_grasp]),
        np.array([1.75], dtype=np.float32),
    )
    # holding the tool has to be worth more than reaching the target without it
    assert (
        wrapped.compute_reward(achieved, desired, [valid])[0]
        > wrapped.compute_reward(achieved, desired, [no_grasp])[0]
    )
    np.testing.assert_allclose(
        wrapped.compute_reward(achieved, desired, [dropped]),
        np.array([-25.0], dtype=np.float32),
    )

    no_progress = np.array([[0.0]], dtype=np.float32)
    partial_progress = np.array([[0.5 * target]], dtype=np.float32)
    empty_grasp = {**valid, "grasp_contact_count": 0}
    reward_at_zero = wrapped.compute_reward(no_progress, desired, [empty_grasp])
    reward_at_half_target = wrapped.compute_reward(
        partial_progress, desired, [empty_grasp]
    )
    assert reward_at_half_target[0] > reward_at_zero[0]

    with_thumb = {**no_grasp, "thumb_on_plunger": True}
    thumb_reward = wrapped.compute_reward(achieved, desired, [with_thumb])[0]
    no_thumb_reward = wrapped.compute_reward(achieved, desired, [no_grasp])[0]
    assert thumb_reward > no_thumb_reward
    assert wrapped.failure_penalty == pytest.approx(25.0)

    _, step_reward, _, _, step_info = wrapped.step(
        np.zeros(wrapped.action_space.shape, dtype=np.float32)
    )
    assert step_reward == pytest.approx(step_info["reward_total"])
    assert step_info["reward_tracking"] >= 0.0
    assert np.isfinite(step_info["reward_push_progress"])
    assert -1.0 <= step_info["reward_push_progress"] <= 1.0
    assert step_info["reward_grasp"] >= 0.0
    wrapped.close()


def test_push_and_spray_reward_share_the_same_shaping() -> None:
    """Both tasks must be trained under the same reward to be compared.

    The push wrapper used to ramp its tracking term over the whole travel and
    pay it whether or not the object was held, while spray peaked at the target
    and gated on the grasp. Comparing synergies across tasks trained that way
    would be comparing the rewards as much as the tasks.
    """
    import numpy as np

    from improved_tds.environments.pose_pool import PushGoalWrapper, SprayGoalWrapper

    for wrapper in (PushGoalWrapper, SprayGoalWrapper):
        source = wrapper._reward_terms.__code__.co_consts
        assert any(
            isinstance(value, str) and "grasp_gate" in value
            for value in wrapper.__init__.__code__.co_names
        ) or "grasp_gate_floor" in wrapper.__init__.__code__.co_varnames, wrapper.__name__
        del source

    # The shared broad component supplies a far-target signal while the fine
    # component still peaks tightly at the target.
    class _Stub:
        grasp_gate_floor = 0.0
        required_grasp_contacts = 2
        tracking_reward_weight = 1.0
        push_progress_reward_weight = 0.0
        grasp_reward_weight = 0.0
        thumb_contact_reward_weight = 0.0
        success_bonus = 0.0
        failure_penalty = 1.0

        class env:
            class unwrapped:
                push_limit = 1.0

    info = {
        "push_limit": 1.0,
        "push_tracking_tolerance": 0.05,
        "grasp_contact_count": 2,
        "thumb_on_plunger": False,
    }
    on_target = PushGoalWrapper._reward_terms(
        _Stub, np.array([0.5]), np.array([0.5]), [info]
    )["tracking"]
    off_target = PushGoalWrapper._reward_terms(
        _Stub, np.array([0.9]), np.array([0.5]), [info]
    )["tracking"]
    assert float(on_target[0]) > 0.9
    assert 0.01 < float(off_target[0]) < 0.3


def test_tool_categories_share_common_reward_defaults() -> None:
    """Task-specific wrappers keep comparable common-term scales."""
    import inspect

    defaults = {
        wrapper: {
            name: parameter.default
            for name, parameter in inspect.signature(wrapper).parameters.items()
        }
        for wrapper in (PushGoalWrapper, SprayGoalWrapper, ScissorsGoalWrapper)
    }
    assert defaults[PushGoalWrapper]["tracking_reward_weight"] == 2.0
    assert defaults[SprayGoalWrapper]["tracking_reward_weight"] == 2.0
    assert (
        defaults[ScissorsGoalWrapper]["coarse_tracking_reward_weight"]
        + defaults[ScissorsGoalWrapper]["fine_tracking_reward_weight"]
        == 2.0
    )
    for wrapper, progress_name, interaction_name in (
        (PushGoalWrapper, "push_progress_reward_weight", "thumb_contact_reward_weight"),
        (SprayGoalWrapper, "trigger_progress_reward_weight", "index_proximity_reward_weight"),
        (ScissorsGoalWrapper, "progress_reward_weight", "interaction_reward_weight"),
    ):
        assert defaults[wrapper][progress_name] == 1.0
        assert defaults[wrapper]["grasp_reward_weight"] == 1.0
        assert defaults[wrapper][interaction_name] == 0.25
        assert defaults[wrapper]["success_bonus"] == 5.0
        assert defaults[wrapper]["failure_penalty"] == 25.0
        assert defaults[wrapper]["grasp_gate_floor"] == 0.25
        assert defaults[wrapper]["control_delta_penalty_weight"] == 0.1
        assert defaults[wrapper]["control_magnitude_penalty_weight"] == 0.002
