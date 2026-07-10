# TDS mathematics and assumptions

Let joint posture be `q` in radians, mean posture be `q_bar`, unit actuation direction
be `s_a`, and one-dimensional coordinate be `rho`:

`q = q_bar + s_a rho`.

## Estimators

PCA-TDS uses the first right singular vector of centered successful samples. Its
explained variance ratio is retained as a baseline metric. Supervised TDS standardizes
joint samples, then uses either one-component PLS or joint/tool-state covariance. The
learned vector is converted back to physical joint space and normalized. When tool state
is available, its sign is chosen so `corr(rho, c) >= 0`.

For instance directions `s_j`, family TDS uses the principal eigenvector of
`sum_j w_j s_j s_j^T` after sign alignment. Each held-out instance keeps a separate
mean posture and calibration.

## Tool-state calibration

`c=f(rho)` is represented by a monotone linear, piecewise-linear, isotonic or PCHIP
curve. Duplicate coordinates are averaged, non-invertible flat intervals are removed,
and inverse extrapolation is disabled by default. Open and close phases may have
separate curves; their mean overlap difference is the reported hysteresis.

## Feedback and generalized force

Tool-state feedback is

`rho_cmd = f^-1(c_d) + Kp e + Ki integral(e) + Kd filtered(d e/dt)`.

Integral action defaults to zero and uses conditional anti-windup. The generalized
TDS reaction estimate is `eta_rho = s_a^T tau`. Admittance is discretized with
semi-implicit Euler:

`M rho_ddot + D rho_dot = Kc(c_d-c) + Keta(eta_d-eta_rho)`.

Low-pass filtering, position/rate/acceleration saturation and force dropout fallback
are mandatory. With one actuation DoF, position and reaction force cannot be controlled
independently; `eta_d` is a compliance/safety preference, not a guaranteed force target.

## Stabilization

The command is `q_cmd = q_bar + s_a rho + b_g phi`. A PCA nullspace is not assumed to
be an internal-force subspace. Manual/PCA directions require an interference metric;
the experimental estimator projects a force-sensitive direction away from measured
tool-state and actuation gradients. `phi`, force, tool deviation and joint commands are
bounded, with emergency release on configured violations.

