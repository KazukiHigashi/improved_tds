# Controllers and safety

All controllers accept `ControlObservation` and return a joint command, TDS coordinate,
mode, saturation flag and diagnostics. `SafetyLimiter` applies rho range, rate,
acceleration and joint limits. Torque, current, force or external emergency-stop
violations return the estimator mean posture as an emergency-release command.

- `FeedforwardTDSController`: inverse calibration and TDS decode baseline.
- `ToolStateFeedbackController`: filtered PID; invalid tool sensor degrades to feedforward.
- `AdmittanceTDSController`: semi-implicit compliant motion; absent force estimate
  degrades to tool-state feedback.
- `StabilizationSynergy`: a separately limited `b_g phi` component that weakens when
  tool-state deviation grows and releases at the force limit.

Safety values in example YAML are simulation defaults, not validated real-hardware
limits. Real limits must be provided by the platform owner before commands are enabled.

