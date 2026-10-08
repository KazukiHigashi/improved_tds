"""Test whether authored initial hand posture predicts a transferable TDS offset.

The frozen 15-task success-only posture databases are used. No task ID is a
numeric feature, no new RL is trained, and leave-one-tool-out fits never see
postures from the target tool. A single operation axis is fitted to within-tool
angle variation. Two rank-one residual corrections are compared:

* ``op_projection``: only the initial posture's operation-axis score (user idea)
* ``full_initial``: one PLS2 latent score from all 13 initial joint coordinates

Each initial pose contributes only one averaged residual, avoiding pseudo-
replication from the several target angles collected at that reset pose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression


CATEGORIES = ("scissors", "spray", "push")


def unit(a):
    a = np.asarray(a, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(a))
    if norm < 1e-10:
        raise ValueError("degenerate direction")
    return a / norm


def read_category(bundle: Path, category: str):
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    result = []
    for variant in range(1, 6):
        name = f"{category}_lite_{variant}"
        file = bundle / name / "database" / "success_postures.npz"
        expected = manifest["tasks"][name]["database_sha256"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} is not the frozen success database")
        policy = bundle / name / "training" / "best_success_policy.zip"
        if (not policy.is_file() or hashlib.sha256(policy.read_bytes()).hexdigest()
                != manifest["tasks"][name]["best_success_policy_sha256"]):
            raise ValueError(f"{name}: frozen best policy missing or changed")
        with np.load(file, allow_pickle=False) as db:
            q = np.asarray(db["q"], float)
            q0 = np.asarray(db["initial_q"], float)
            pose = np.asarray(db["pose_index"], int)
            y = np.asarray(db["tool_state"], float) / float(np.max(db["target"]))
            if (q.shape != (200, 13) or q0.shape != q.shape or
                    not np.all(db["success"]) or not np.all(np.isfinite(q)) or
                    not np.all(np.isfinite(q0)) or not np.all(np.isfinite(y))):
                raise ValueError(f"{name}: incompatible success DB")
        result.append(dict(name=name, q=q, q0=q0, y=y, pose=pose))
    return result


def fit_operation(blocks):
    q = np.concatenate([b["q"] for b in blocks])
    x = np.concatenate([b["q"] - b["q"].mean(0) for b in blocks])
    y = np.concatenate([b["y"] - b["y"].mean() for b in blocks])
    axis = unit(PLSRegression(n_components=1, scale=False).fit(x, y).x_weights_[:, 0])
    if float((x @ axis) @ y) < 0:
        axis *= -1
    return q.mean(0), axis


def pose_means(blocks, final_mean, operation):
    """One paired initial/resulting residual per authored reset pose."""
    entries = []
    for b in blocks:
        for pose in np.unique(b["pose"]):
            rows = b["pose"] == pose
            q0 = b["q0"][rows]
            if np.max(np.abs(q0 - q0[0])) > 1e-6:
                raise ValueError(f"{b['name']} pose {pose} has inconsistent initial q")
            qbar = b["q"][rows].mean(0)
            residual = qbar - final_mean
            residual -= float(residual @ operation) * operation
            entries.append(dict(task=b["name"], pose=int(pose),
                                q0=q0[0], residual=residual,
                                sample_count=int(rows.sum())))
    return entries


def fit_correction(entries, final_mean, operation, method):
    q0 = np.stack([e["q0"] for e in entries])
    residual = np.stack([e["residual"] for e in entries])
    q0_mean = q0.mean(0)
    x0 = q0 - q0_mean
    r_mean = residual.mean(0)
    r = residual - r_mean
    if method == "op_projection":
        w = operation.copy()
        predictor = x0 @ w
        v2 = unit(r.T @ predictor)
    elif method == "full_initial":
        # Leading PLS2 cross-covariance mode, not a task-ID direction.
        w, _, vh = np.linalg.svd(x0.T @ r, full_matrices=False)
        w = unit(w[:, 0])
        v2 = unit(vh[0])
        predictor = x0 @ w
    else:
        raise ValueError(method)
    v2 = unit(v2 - float(v2 @ operation) * operation)
    score = residual @ v2
    design = np.column_stack([np.ones(len(predictor)), predictor])
    coefficients = np.linalg.lstsq(design, score, rcond=None)[0]
    fitted_g = design @ coefficients
    return dict(method=method, final_mean=final_mean, q0_mean=q0_mean,
                operation=operation, initial_loading=w, correction_axis=v2,
                coefficients=coefficients,
                g_limits=np.array([min(0., 1.5 * fitted_g.min()),
                                   max(0., 1.5 * fitted_g.max())]))


def predict_correction(model, initial_q):
    q0 = np.asarray(initial_q, float)
    scalar = (q0 - model["q0_mean"]) @ model["initial_loading"]
    g = model["coefficients"][0] + model["coefficients"][1] * scalar
    return np.asarray(g, float)


def evaluate_models(train, test, tag):
    center, operation = fit_operation(train)
    training_poses = pose_means(train, center, operation)
    testing_poses = pose_means(test, center, operation)
    q0_test = np.stack([e["q0"] for e in testing_poses])
    r_test = np.stack([e["residual"] for e in testing_poses])
    base_mse = float(np.mean(np.sum(r_test**2, axis=1)))
    models = {}
    predictions = {}
    metrics = {}
    for method in ("op_projection", "full_initial"):
        model = fit_correction(training_poses, center, operation, method)
        g = predict_correction(model, q0_test)
        v = model["correction_axis"]
        errors = np.sum((r_test - g[:, None] * v)**2, axis=1)
        oracle_g = r_test @ v
        oracle_errors = np.sum((r_test - oracle_g[:, None] * v)**2, axis=1)
        models[method] = model
        predictions[method] = g
        metrics[method] = dict(
            residual_mse=float(np.mean(errors)),
            improvement_vs_zero=float(1 - np.mean(errors) / base_mse),
            oracle_axis_improvement=float(1 - np.mean(oracle_errors) / base_mse),
            correction_score_correlation=(float(np.corrcoef(g, oracle_g)[0, 1])
                if np.std(g) > 1e-12 and np.std(oracle_g) > 1e-12 else None),
            mean_abs_correction_score=float(np.mean(np.abs(g))),
            correction_axis_operation_dot=float(v @ operation),
        )
    train_q0 = np.stack([e["q0"] for e in training_poses])
    distances = np.linalg.norm(q0_test[:, None, :] - train_q0[None, :, :], axis=2)
    test_nearest = distances.min(axis=1)
    same = np.linalg.norm(train_q0[:, None, :] - train_q0[None, :, :], axis=2)
    np.fill_diagonal(same, np.inf)
    train_nearest = same.min(axis=1)
    train_scale = float(np.median(train_nearest))
    diagnostics = dict(
        protocol=tag,
        training_tools=[b["name"] for b in train],
        test_tools=[b["name"] for b in test],
        train_pose_count=len(training_poses), test_pose_count=len(testing_poses),
        baseline_residual_mse=base_mse,
        initial_nn_distance_median=float(np.median(test_nearest)),
        initial_nn_distance_to_training_ratio=float(np.median(test_nearest) / train_scale),
        operation_axis=operation.tolist(), final_mean=center.tolist(),
        operation_axis_cosine_to_heldout=(float(operation @ fit_operation(test)[1])
            if len(test) == 1 else None),
        methods=metrics,
        rows=[dict(task=e["task"], pose=e["pose"],
                   initial_operation_score=float((e["q0"] - center) @ operation),
                   initial_nn_distance=float(test_nearest[i]),
                   residual_norm=float(np.linalg.norm(e["residual"])),
                   **{f"{method}_predicted_g":float(predictions[method][i])
                      for method in predictions},
                   **{f"{method}_oracle_g":float(e["residual"] @ models[method]["correction_axis"])
                      for method in models})
              for i, e in enumerate(testing_poses)]
    )
    return diagnostics, models


def to_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def mean_metric(items, method, key):
    return float(np.mean([x["methods"][method][key] for x in items]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path,
                   default=Path("analysis/paper15_paper_bundle_20260928"))
    p.add_argument("--output", type=Path,
                   default=Path("analysis/initial_pose_conditioned_tds_20261007"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(protocol="initial_pose_conditioning_diagnostic_v1",
                  source=str(args.bundle.resolve()), new_training=False,
                  source_policy="frozen per-task best_success_policy.zip and success-only DB",
                  interpretation="one fixed operation axis; one initial-pose-predicted residual coordinate",
                  categories={})
    for category in CATEGORIES:
        blocks = read_category(args.bundle, category)
        loso = []
        for i in range(5):
            train = [b for j, b in enumerate(blocks) if j != i]
            result, _ = evaluate_models(train, [blocks[i]], "leave_one_tool_out")
            loso.append(result)
        held_pose = []
        for seed in range(10):
            rng = np.random.default_rng(78410 + seed)
            train = []
            test = []
            for b in blocks:
                ids = np.unique(b["pose"])
                selected = rng.choice(ids, size=max(1, round(.25 * len(ids))), replace=False)
                mask = np.isin(b["pose"], selected)
                train.append({**b, **{key:b[key][~mask] for key in ("q", "q0", "y", "pose")}})
                test.append({**b, **{key:b[key][mask] for key in ("q", "q0", "y", "pose")}})
            result, _ = evaluate_models(train, test, "held_out_initial_poses")
            held_pose.append(result)
        report["categories"][category] = dict(
            leave_one_tool_out=loso, held_out_initial_poses=held_pose,
            summary={phase:{method:{key:mean_metric(items, method, key)
                for key in ("improvement_vs_zero", "oracle_axis_improvement",
                            "mean_abs_correction_score")}
                for method in ("op_projection", "full_initial")}
                for phase, items in (("leave_one_tool_out", loso),
                                     ("held_out_initial_poses", held_pose))},
            median_initial_nn_distance_ratio=float(np.median(
                [x["initial_nn_distance_to_training_ratio"] for x in loso])))
        print(category, json.dumps(report["categories"][category]["summary"]), flush=True)
    (args.output / "offline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=to_json), encoding="utf-8")


if __name__ == "__main__":
    main()
