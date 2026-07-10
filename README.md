# Improved TDS

`improved-tds` is a standalone research project for low-dimensional, closed-loop
actuation of a known or initialized grasp on an articulated 1-DoF tool. It evolves the
`tool-dof-synergy` PCA/feedforward baseline without modifying that reference project.

Implemented features include versioned successful-trajectory datasets, PCA and
supervised/family TDS estimators, monotone few-shot calibration, feedforward/PID/
admittance controllers, independent stabilization synergy, force/hardware interfaces,
and Gymnasium/MuJoCo environments for scissors, triggers and buttons.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

The numerical core is CPU-only and does not require Gymnasium, MuJoCo, SB3 or a GPU.
Install `.[simulation]` for environments or `.[training]` for the optional SB3 DDPG+HER
baseline. No Isaac Sim or other heavyweight simulator is used.

## Compatibility and environments

The copied MIT-licensed scissors assets are self-contained. Existing IDs
`ExpScissor1-v0` through `ExpScissor5-v0` keep the HER Dict observation and normalized
14-actuator action contract. New IDs are `TDS-Trigger-v0` and `TDS-Button-v0`.

```python
import gymnasium as gym
import improved_tds

env = gym.make("TDS-Trigger-v0")
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

The end-to-end data/fitting commands are documented in
[`docs/experiments.md`](docs/experiments.md). Architecture survey and compatibility
decisions are in
[`docs/implementation_plan_closed_loop_tds.md`](docs/implementation_plan_closed_loop_tds.md).

## Safety boundary

Simulation defaults are not real-hardware limits. Motor constants, sensor calibration,
command transport and safety limits are deliberately not guessed. Real integration
must implement `HardwareAdapter`, supply validated configuration and retain emergency
release/watchdog behavior described in
[`docs/real_robot_integration.md`](docs/real_robot_integration.md).

Scissors assets retain their upstream notice in
[`src/improved_tds/assets/LICENSE.md`](src/improved_tds/assets/LICENSE.md).

