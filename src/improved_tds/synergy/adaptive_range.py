"""Calibration-only direction and interval selection for scalar synergy control."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def amplitude(axis, low, high):
    axis = np.asarray(axis, dtype=float)
    active = np.abs(axis) > 1e-12
    return float(np.min((np.asarray(high) - low)[active] / 2 / np.abs(axis[active])))


def posture(grasp, axis, scale, rho, low, high):
    """rho is in units of the original nominal amplitude, not bounded to [-1,1]."""
    return np.clip(grasp + float(rho) * scale * axis, low, high)


def fit_ramp(u, tool):
    """Fit a monotone, clipped linear ramp on the same calibration means as isotonic.

    Ramp breakpoints are selected by deterministic grid search; the two output
    levels are least-squares estimates. No test response is used.
    """
    u=np.asarray(u,dtype=float)
    tool=np.asarray(tool,dtype=float)
    if u.ndim!=1 or tool.shape!=u.shape or len(u)<3 or not np.isfinite(tool).all():
        raise ValueError('ramp needs at least three finite calibration pairs')
    grid=np.linspace(float(u[0]),float(u[-1]),41)
    best=None
    for start in grid[:-1]:
        for end in grid[grid>start]:
            z=np.clip((u-start)/(end-start),0.,1.)
            design=np.column_stack((np.ones_like(z),z))
            lower,gain=np.linalg.lstsq(design,tool,rcond=None)[0]
            if gain<=1e-9:
                continue
            predicted=lower+gain*z
            sse=float(np.sum((predicted-tool)**2))
            candidate=(sse,float(start),float(end),float(lower),float(lower+gain))
            if best is None or candidate<best:
                best=candidate
    if best is None:
        return None
    sse,start,end,lower,upper=best
    return dict(start=start,end=end,tool_low=lower,tool_high=upper,
                fitted=(lower+(upper-lower)*np.clip((u-start)/(end-start),0.,1.)).tolist(),
                calibration_rmse=float(np.sqrt(sse/len(u))))


def invert_ramp(ramp, target):
    """Return a ramp score, clipping only to the fitted response range."""
    fraction=np.clip((float(target)-ramp['tool_low'])/(ramp['tool_high']-ramp['tool_low']),0.,1.)
    return float(ramp['start']+fraction*(ramp['end']-ramp['start']))


def common_inverse_domain(selection, target_range):
    """Calibration-only target interval supported by both fitted inverses.

    Scores outside this interval are not labelled as inverse-control tests:
    both fits would merely clamp them to an endpoint.  This intentionally
    retains the original RL target range as a separate coverage denominator.
    """
    if selection is None or selection.get('ramp') is None:
        return None
    low=max(float(target_range[0]),float(selection['map_tool'][0]),
            float(selection['ramp']['tool_low']))
    high=min(float(target_range[1]),float(selection['map_tool'][-1]),
             float(selection['ramp']['tool_high']))
    return [low,high] if high>low+1e-9 else None


def select_interval(x, responses, valid, target_range, *, full=False):
    """Require a fixed held cohort, appreciable range and consistent direction.

    Thresholds are preregistered engineering choices, not claims of significance.
    Test responses must never be passed to this function.
    """
    x, y, valid = np.asarray(x), np.asarray(responses), np.asarray(valid, dtype=bool)
    width = float(target_range[1] - target_range[0])
    candidates = []
    pairs = [(0, len(x) - 1)] if full else [
        (i, j) for i in range(len(x) - 2) for j in range(i + 2, len(x))
    ]
    for i, j in pairs:
        cohort = valid[:, i:j + 1].all(axis=1)
        held = float(cohort.mean())
        if held < 0.75:
            continue
        values = y[cohort, i:j + 1]
        direction = 1 if np.mean(values[:, -1] - values[:, 0]) >= 0 else -1
        agreement = float(np.mean(direction * (values[:, -1] - values[:, 0]) > .01 * width))
        means = values.mean(axis=0)
        gain = direction * float(means[-1] - means[0])
        if agreement < .75 or gain < .05 * width:
            continue
        xx = x[i:j + 1][::direction]
        yy = means[::direction]
        drops = np.maximum(-np.diff(yy), 0)
        if not full and (np.max(drops) > .02 * width or np.sum(drops) > .05 * width):
            continue
        u = -1. + 2. * (xx - xx[0]) / (xx[-1] - xx[0])
        fitted = IsotonicRegression(increasing=True).fit_transform(u, yy)
        ramp = fit_ramp(u, yy)
        # Average coordinates on plateaus; no arbitrary first-point inversion.
        unique = np.unique(fitted)
        map_u = [float(u[fitted == value].mean()) for value in unique]
        coverage = max(0., min(target_range[1], unique[-1]) - max(target_range[0], unique[0])) / width
        if coverage < .05:
            continue
        targets = np.linspace(*target_range, 9)
        requested_u = np.interp(targets, unique, map_u)
        calibration_mae = float(np.mean([
            np.abs(np.interp(requested_u, u, curve[::direction]) - targets).mean()
            for curve in values
        ]) / width)
        candidates.append(dict(
            rho_start=float(xx[0]), rho_end=float(xx[-1]), direction=direction,
            coverage=float(coverage), held_rate=held, direction_agreement=agreement,
            reverse_travel=float(np.sum(drops) / width), map_u=map_u,
            isotonic_calibration_rmse=float(np.sqrt(np.mean((fitted-yy)**2))),
            calibration_interpolated_normalised_mae=calibration_mae,
            map_tool=unique.tolist(), raw_rho=xx.tolist(), raw_tool=yy.tolist(),
            ramp=ramp,
            cohort_indices=np.flatnonzero(cohort).tolist(),
        ))
    if not candidates:
        return None
    return max(candidates, key=lambda c: (
        c['coverage'], c['held_rate'], c['direction_agreement'],
        -c['calibration_interpolated_normalised_mae'], -c['reverse_travel'],
        -abs(c['rho_end'] - c['rho_start']),
    ))
