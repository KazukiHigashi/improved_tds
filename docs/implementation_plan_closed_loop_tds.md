# Closed-loop TDS implementation plan and repository survey

## Survey of `tool-dof-synergy`

Only `/home/synergy/mujoco/projects/tool-dof-synergy` was used as the reference.
The executable modern port is the outer `tool_dof_synergy/` package; the nested
`tool_dof_synergy-main/` tree is the historical TensorFlow/OpenAI Baselines source.

- Environments: `ExpScissorEnv` registers `ExpScissor1-v0` through `5-v0`. It uses
  Gymnasium's five-value `step` API and HER-compatible Dict observations.
- MuJoCo: five scissors XMLs include a 14-actuator Shadow Hand Lite, two scissors
  hinge joints and touch sensors. The modern port uses the official `mujoco` API.
- Observation/action: observation has shape `(67,)`; achieved/desired goal have
  shape `(1,)`; normalized action has shape `(14,)` and maps to actuator ctrl ranges.
- RL: the modern path is SB3 DDPG/SAC/TD3, with DDPG defaults chosen to approximate
  historical HER-DDPG. The historical replay buffer stores whole episodes and HER
  samples future goals. The modern custom HER buffer recomputes PCA-dependent reward.
- Existing synergy data: `SynergyManager` adds successful late-episode action or
  posture samples and persists an object-array `synergy_dataset.npy` containing
  `[postures, scissors_angles]`. It fits multi-component sklearn PCA and uses
  reconstruction error as reward; it is not a trajectory logger.
- Calibration: no independent reusable `c=f(rho)` object exists in the modern port.
  Historical visualization/real scripts contain local fitting and ramp functions.
- Real hardware: historical ROS scripts send Shadow Hand joint trajectories and a
  serial reader returns scissors angle in radians. They do not expose a stable torque,
  current, tool sensor or emergency-stop adapter API.
- Tests/config: the reference has smoke/train/evaluate scripts and resolved JSON run
  configs, but no repository test suite for TDS math or controller safety.

## Adopted structure

The new `improved_tds` package separates pure numerical core modules from optional
simulation/training boundaries:

```text
data/          trajectory schema, validation, legacy converter
synergy/       PCA, supervised, family TDS, calibration
control/       feedforward, feedback, admittance, stabilization, safety
tools/         tool model, force estimators, real/mock/replay interfaces
environments/  articulated API, legacy-compatible scissors, trigger, button
learning/      successful full-trajectory collector
evaluation/    tracking and leave-one-instance-out metrics
experiments/   CLI entry points; SB3 remains optional
```

## Changed and new files

All files are new in `improved-tds`; no reference-project file is modified. Licensed
scissors XML/STL/texture assets and the modern environment code were copied, isolated
as `legacy_scissors.py`, then wrapped by the articulated-tool API. See `git status`
for the authoritative file list.

## Compatibility and risk decisions

1. `ExpScissor1-v0` through `5-v0`, Dict keys, `(67,)` observation and `(14,)`
   action are retained. New names `ImprovedScissor1-v0` through `5-v0` are aliases.
2. Legacy object arrays are accepted only by an explicit converter using
   `allow_pickle=True`; the new schema never requires pickle and rejects ragged/NaN data.
3. PCA is kept as baseline. All one-dimensional estimators normalize physical joint
   direction and align sign to positive tool-state correlation.
4. Gymnasium, MuJoCo and SB3 are optional boundaries. Core estimation/control can be
   imported without them. SB3 DDPG+HER is an optional data-generation baseline.
5. Scissors geometry scale and unknown real hardware values are not silently changed.
   Runtime parameters are limited to quantities MuJoCo can safely update.

## Unknown hardware dependencies

Motor torque constants, gear ratios, current and torque signs, sensor timestamp/latency,
tool sensor calibration, controller transport, joint/force/current limits, emergency
stop semantics and real tool geometry are not supplied. Real actuation is intentionally
limited to protocol, mock and offline replay implementations.

## Implementation and test order

The order is schema/interfaces, TDS and calibration, safety/controllers, environments,
collection/evaluation, then hardware integration. Fast unit tests cover math and
serialization; simulation tests cover all seven registered environments; optional
training/long experiments remain separately marked.

