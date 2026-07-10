# Trajectory datasets

`TrajectoryStep` stores synchronized timestamp, joint position/velocity, action,
tool state/rate, optional torque/current/contact data, phase, reward and episode flags.
`TrajectoryMetadata` stores family, instance, task, success, seed, simulator parameters,
controller and checkpoint. New `.npz` files include `schema_version=1.0`, episode offsets,
JSON metadata and dense arrays; they do not contain pickled Python objects.

Validation rejects empty data, NaN/infinity, non-monotone timestamps, inconsistent
feature dimensions and mismatched field lengths. `SuccessfulTrajectoryCollector`
retains full successful episodes by default. Failed episodes can be kept explicitly for
debugging, but `samples()` returns only successful trajectories.

Legacy files are object arrays `[postures, tool_angles]`. Convert explicitly:

```bash
python -m improved_tds.experiments.convert_legacy \
  synergy_dataset.npy trajectories.npz --instance-id scissors-1
```

Legacy terminal samples have no velocity/contact history; the converter marks this in
metadata and creates one-step terminal trajectories rather than inventing measurements.

