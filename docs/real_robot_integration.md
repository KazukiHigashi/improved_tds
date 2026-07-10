# Real robot integration boundary

`HardwareAdapter` separates `read_state`, `write_command`, emergency stop and close.
`MockHardwareAdapter` and `OfflineReplayAdapter` allow the same controller loop to run
without hardware. `ForceEstimator` implementations support direct joint torque,
explicit motor-current conversion, PD tracking-error proxy and deterministic mocks.

Before adding a spray-bottle or remote-control adapter, provide and validate:

1. joint names/order, radian conventions and command transport;
2. timestamp/clock synchronization and maximum sensor latency;
3. tool-state sensor units, range, sign and dropout behavior;
4. torque/current sign, motor torque constants and gear ratios;
5. joint, rate, acceleration, current, torque and contact-force limits;
6. emergency-stop/release semantics and a hardware watchdog;
7. tool geometry and an independently reviewed initial grasp.

No real command should be enabled by simulation defaults. Bias reset must occur in a
known unloaded condition, stale observations must enter dropout fallback, and every log
must retain raw sensor values alongside controller diagnostics.

