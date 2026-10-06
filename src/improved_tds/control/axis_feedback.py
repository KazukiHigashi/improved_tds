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

    def __post_init__(self):
        values = vars(self)
        if any(not np.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError('feedback settings must be positive and finite')
        if self.probe_offset > self.max_offset:
            raise ValueError('probe offset exceeds allowed offset')


def estimate_sensitivity(probes, offset, tool_span, noise_fraction=0.001):
    """Paired minus/zero/plus probes, one triplet per CALIBRATION pose.

    Estimate each side separately. An invalid probe removes only that side;
    contact-link changes are logged by the experiment but do not invalidate
    tool response. A side needs at least two usable poses with a consistent
    sign and response above the measured within-hold noise. Poses with the
    minority response sign are excluded from that side and counted. Opposite-sided
    slopes with opposite signs describe a turning point: use only the side
    with more support, then the stronger response as a deterministic tie-break.
    """
    if not np.isfinite(offset) or offset <= 0 or tool_span <= 0:
        raise ValueError('positive probe offset and tool span required')
    sides = {'minus': [], 'plus': []}
    exclusions = []
    for pose, triplet in enumerate(probes):
        if len(triplet) != 3:
            raise ValueError('each probe must contain minus, zero, plus')
        zero = triplet[1]
        for label, index in (('minus', 0), ('plus', 2)):
            other = triplet[index]
            samples = (zero, other)
            if not all(p['valid'] and p.get('complete', False)
                       and not p.get('clipped', False) for p in samples):
                exclusions.append({'pose': pose, 'side': label, 'reason': 'invalid_or_clipped_probe'})
                continue
            if not all(np.isfinite(p['terminal_mean']) and np.isfinite(p['terminal_std'])
                       for p in samples):
                exclusions.append({'pose': pose, 'side': label, 'reason': 'nonfinite_probe'})
                continue
            noise = max(tool_span * noise_fraction,
                        3 * max(float(p['terminal_std']) for p in samples))
            slope = ((float(zero['terminal_mean']) - float(other['terminal_mean'])) / offset
                     if label == 'minus' else
                     (float(other['terminal_mean']) - float(zero['terminal_mean'])) / offset)
            if abs(slope) * offset <= noise:
                exclusions.append({'pose': pose, 'side': label, 'reason': 'response_below_noise'})
                continue
            sides[label].append({'pose': pose, 'slope': slope, 'noise': noise})
    usable = {}
    for label, samples in sides.items():
        positive = [s for s in samples if s['slope'] > 0]
        negative = [s for s in samples if s['slope'] < 0]
        if positive and negative and len(positive) == len(negative):
            exclusions.append({'side': label, 'reason': 'response_sign_tie'})
            continue
        selected_sign = max((positive, negative), key=lambda group:
                            (len(group), sum(abs(s['slope']) for s in group)))
        if len(selected_sign) < 2:
            continue
        for removed in samples:
            if removed not in selected_sign:
                exclusions.append({'pose': removed['pose'], 'side': label,
                                   'reason': 'opposite_response_sign'})
        usable[label] = selected_sign
    if len(usable) == 2:
        a, b = (np.median([s['slope'] for s in usable[label]])
                for label in ('minus', 'plus'))
        if a * b < 0:
            rejected = min(usable, key=lambda name:
                (len(usable[name]), np.median([abs(s['slope']) for s in usable[name]])))
            exclusions.append({'side': rejected, 'reason': 'turning_point_restrict_to_one_side'})
            del usable[rejected]
    selected = [s for samples in usable.values() for s in samples]
    base = dict(usable_poses=len({s['pose'] for s in selected}), total_poses=len(probes),
                excluded_reasons=exclusions,
                usable_sides={k: len(v) for k, v in usable.items()},
                noise=float(np.median([s['noise'] for s in selected])) if selected
                else tool_span * noise_fraction,
                offset_low=-offset if 'minus' in usable else 0.,
                offset_high=offset if 'plus' in usable else 0.,
                enabled=False, slope=None)
    if not selected:
        return dict(base, reason='no_consistent_measurable_side')
    return dict(base, enabled=True,
                slope=float(np.median([s['slope'] for s in selected])),
                reason='calibrated_both_sides' if len(usable) == 2 else 'calibrated_one_side')


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
