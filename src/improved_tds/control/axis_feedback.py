"""Conservative scalar feedback in a calibrated synergy coordinate.

Independent of MuJoCo. No gain fitting, probing or target-dependent tuning is
performed on evaluation poses. A latched stop holds the last command; it is not
a real-hardware emergency-stop implementation.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxisFeedbackConfig:
    gain_per_second: float = 1.0
    rate_per_second: float = 0.25
    max_increment: float = 0.05
    max_offset: float = 0.5
    probe_offset: float = 0.1
    deadband_fraction: float = 0.005
    noise_fraction: float = 0.001
    contact_debounce_seconds: float = 0.1

    def __post_init__(self):
        values = vars(self)
        if any(not np.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError('feedback settings must be positive and finite')
        if self.probe_offset > self.max_offset:
            raise ValueError('probe offset exceeds allowed offset')


def estimate_sensitivity(probes, offset, tool_span, noise_fraction=0.001):
    """Paired minus/zero/plus probes, one triplet per CALIBRATION pose.

    Use two one-sided slopes, not only the central difference (which hides
    turning points). Require consistent signs across every usable pose and
    both sides, and at least two usable poses. Within-hold SD is deliberately
    not divided by sqrt(n), since simulation samples are serially correlated.
    """
    if not np.isfinite(offset) or offset <= 0 or tool_span <= 0:
        raise ValueError('positive probe offset and tool span required')
    slopes, noises, reasons, loads = [], [], [], []
    for triplet in probes:
        if len(triplet) != 3:
            raise ValueError('each probe must contain minus, zero, plus')
        if not all(p['valid'] and p.get('complete', False)
                   and not p.get('clipped', False) for p in triplet):
            reasons.append('invalid_or_clipped_probe')
            continue
        signatures = [p.get('terminal_contact_signature') for p in triplet]
        if any(s != signatures[1] for s in signatures):
            reasons.append('probe_contact_mode_changed')
            continue
        y = np.array([p['terminal_mean'] for p in triplet], dtype=float)
        noise = max(tool_span * noise_fraction,
                    3 * max(float(p['terminal_std']) for p in triplet))
        if not np.isfinite(y).all() or not np.isfinite(noise):
            reasons.append('nonfinite_probe')
            continue
        slopes.extend((np.diff(y) / offset).tolist())
        noises.append(noise)
        loads.extend(float(p.get('peak_normal_load_n', 0.)) for p in triplet)
    base = dict(usable_poses=len(noises), total_poses=len(probes),
                excluded_reasons=reasons, slopes=slopes, noise=max(noises, default=tool_span*noise_fraction),
                offset_low=-offset, offset_high=offset, enabled=False, slope=None,
                load_limit_n=1.5*max(loads, default=0.) if max(loads,default=0.)>0 else None)
    # Do not label a range safe because only its easiest poses survived.
    if reasons:
        return dict(base, reason='unsafe_or_incomplete_calibration')
    if len(noises) < 2:
        return dict(base, reason='fewer_than_two_usable_calibration_poses')
    slopes = np.asarray(slopes)
    if not (np.all(slopes > 0) or np.all(slopes < 0)):
        return dict(base, reason='nonmonotonic_or_inconsistent_sign')
    if np.min(np.abs(slopes)) * offset <= base['noise']:
        return dict(base, reason='response_below_noise')
    return dict(base, enabled=True, slope=float(np.median(slopes)), reason='calibrated')


class AxisFeedback:
    def __init__(self, sensitivity, tool_span, config=None):
        self.config = config or AxisFeedbackConfig()
        self.sensitivity = sensitivity
        self.span = float(tool_span)
        if not np.isfinite(self.span) or self.span <= 0:
            raise ValueError('positive tool span required')
        self.offset = 0.0
        self.saturated = False
        self.rate_limited = False
        self.stop_reason = None if sensitivity['enabled'] else sensitivity['reason']
        if sensitivity['enabled'] and (not np.isfinite(sensitivity['slope'])
                                      or sensitivity['slope'] == 0):
            raise ValueError('enabled sensitivity needs a finite nonzero slope')

    def stop(self, reason):
        self.stop_reason = self.stop_reason or reason

    def propose(self, target, measured, dt):
        self.saturated = False
        self.rate_limited = False
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('positive finite dt required')
        if not np.isfinite([target, measured]).all():
            self.stop('nonfinite_sensor')
        if self.stop_reason:
            return self.offset
        error = float(target - measured)
        if abs(error) <= max(self.config.deadband_fraction * self.span,
                             self.sensitivity['noise']):
            return self.offset
        delta = self.config.gain_per_second * error / self.sensitivity['slope'] * dt
        limit = min(self.config.max_increment, self.config.rate_per_second * dt)
        lo = max(-self.config.max_offset, self.sensitivity['offset_low'])
        hi = min(self.config.max_offset, self.sensitivity['offset_high'])
        self.rate_limited = bool(abs(delta)>limit)
        proposed = self.offset + np.clip(delta, -limit, limit)
        self.saturated = bool(proposed<=lo or proposed>=hi)
        return float(np.clip(proposed, lo, hi))

    def accept(self, candidate):
        self.offset = float(candidate)
