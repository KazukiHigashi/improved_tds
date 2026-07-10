from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import improved_tds
from improved_tds.environments.randomization import (
    DomainRandomizationConfig,
    DomainRandomizationWrapper,
)
from improved_tds.synergy.random import RandomTDS
from improved_tds.tools.adapter_stubs import SprayBottleAdapterStub
from improved_tds.tools.sensor import MockToolStateSensor


def test_seeded_random_baseline_round_trip(tmp_path) -> None:
    q = np.arange(30, dtype=np.float64).reshape(10, 3)
    state = np.linspace(0.0, 1.0, 10)
    first = RandomTDS(seed=7).fit(q, state)
    second = RandomTDS(seed=7).fit(q, state)
    np.testing.assert_array_equal(first.direction_, second.direction_)
    path = tmp_path / "random.npz"
    first.save(path)
    np.testing.assert_array_equal(RandomTDS.load(path).direction_, first.direction_)


@pytest.mark.simulation
def test_domain_randomization_is_seeded_and_adds_latency_info() -> None:
    improved_tds.register_environments()
    config = DomainRandomizationConfig(
        parameter_ranges={"stiffness": (0.8, 1.2)},
        sensor_noise_std=0.001,
        action_latency_steps=(1, 2),
    )
    env = DomainRandomizationWrapper(gym.make("TDS-Trigger-v0"), config)
    _, first = env.reset(seed=11)
    _, second = env.reset(seed=11)
    assert first["domain_randomization"] == second["domain_randomization"]
    _, _, _, _, info = env.step(np.ones(1, dtype=np.float32))
    np.testing.assert_array_equal(info["applied_action"], np.zeros(1))
    env.close()


def test_sensor_dropout_and_real_stub_fail_closed() -> None:
    sensor = MockToolStateSensor(np.array([0.2]), timestamp=1.0)
    assert sensor.read() is not None
    sensor.dropout = True
    assert sensor.read() is None
    with pytest.raises(RuntimeError, match="stub"):
        SprayBottleAdapterStub().read_state()

