"""Audit 2D initial-pose PCA nearest-neighbour operation-axis transfer.

Task IDs are used only as labels to evaluate neighbour retrieval. PCA and NN
receive only the 13 measured initial hand-joint coordinates. Leave-one-tool-out
fits never include the held-out tool's initial or successful final postures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_initial_pose_conditioning import CATEGORIES, read_category
from simulate_initial_pose_conditioning import fit_local


METHODS = ("raw13", "pca1", "pca2", "pca2_whiten", "pca3")


def unique_initials(blocks):
    samples, labels, pose_ids = [], [], []
    for block in blocks:
        for pose in np.unique(block["pose"]):
            subset = block["q0"][block["pose"] == pose]
            if np.max(np.abs(subset - subset[0])) > 1e-6:
                raise ValueError("one pose has inconsistent initial joint postures")
            samples.append(subset[0])
            labels.append(block["name"])
            pose_ids.append(int(pose))
    return np.stack(samples), np.asarray(labels), np.asarray(pose_ids)


def fit_pca(initials):
    mean = initials.mean(axis=0)
    _, singular, vh = np.linalg.svd(initials - mean, full_matrices=False)
    variance = singular**2 / np.sum(singular**2)
    return dict(mean=mean, components=vh[:3], variance=variance[:3])


def coordinates(initials, model, method):
    if method == "raw13":
        return initials
    dims = {"pca1": 1, "pca2": 2, "pca2_whiten": 2, "pca3": 3}[method]
    score = (initials - model["mean"]) @ model["components"][:dims].T
    if method == "pca2_whiten":
        score = score / np.sqrt(model["variance"][:dims])
    return score


def select_nearest(test_initials, train_initials, model, method):
    test = coordinates(test_initials, model, method)
    train = coordinates(train_initials, model, method)
    distance = np.linalg.norm(test[:, None] - train[None], axis=-1)
    index = np.argmin(distance, axis=1)
    return index, distance[np.arange(len(test)), index]


def heldout_pose_label_accuracy(blocks):
    results = {method:[] for method in METHODS}
    for seed in range(10):
        train, test = [], []
        rng = np.random.default_rng(66520 + seed)
        for b in blocks:
            ids = np.unique(b["pose"])
            chosen = rng.choice(ids, size=max(1, round(.25 * len(ids))), replace=False)
            train.append({**b, **{key:b[key][~np.isin(b["pose"], chosen)]
                                  for key in ("q", "q0", "y", "pose")}})
            test.append({**b, **{key:b[key][np.isin(b["pose"], chosen)]
                                 for key in ("q", "q0", "y", "pose")}})
        x_train, lab_train, _ = unique_initials(train)
        x_test, lab_test, _ = unique_initials(test)
        model = fit_pca(x_train)
        for method in METHODS:
            index, distance = select_nearest(x_test, x_train, model, method)
            results[method].append(dict(seed=seed,
                label_accuracy=float(np.mean(lab_train[index] == lab_test)),
                mean_distance=float(np.mean(distance))))
    return results


def leave_one_tool_out(blocks):
    results = []
    local_axes = {b["name"]:fit_local(b)[1] for b in blocks}
    for heldout, target in enumerate(blocks):
        sources = [b for i,b in enumerate(blocks) if i != heldout]
        train_x, train_labels, train_poses = unique_initials(sources)
        test_x, _, test_poses = unique_initials([target])
        model = fit_pca(train_x)
        entries = dict(heldout_tool=target["name"], train_initial_count=len(train_x),
            test_initial_count=len(test_x),
            variance_ratio=model["variance"].tolist(),
            cumulative_pca2_variance=float(np.sum(model["variance"][:2])),
            methods={})
        raw_idx, _ = select_nearest(test_x, train_x, model, "raw13")
        for method in METHODS:
            nearest_idx, distances = select_nearest(test_x, train_x, model, method)
            chosen_labels = train_labels[nearest_idx]
            selected_cosine = [float(local_axes[label] @ local_axes[target["name"]])
                               for label in chosen_labels]
            entries["methods"][method] = dict(
                selected_source_by_pose={str(int(pose)):str(label)
                                         for pose,label in zip(test_poses, chosen_labels)},
                mean_nearest_distance=float(np.mean(distances)),
                nearest_source_distribution={str(b["name"]):int(np.sum(chosen_labels == b["name"]))
                                             for b in sources},
                mean_operation_axis_cosine=float(np.mean(selected_cosine)),
                min_operation_axis_cosine=float(np.min(selected_cosine)),
                same_neighbour_as_raw13_fraction=float(np.mean(nearest_idx == raw_idx)),
                same_source_tool_as_raw13_fraction=float(np.mean(
                    chosen_labels == train_labels[raw_idx])))
        results.append(entries)
    return results


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path,
                   default=Path("analysis/paper15_paper_bundle_20260928"))
    p.add_argument("--output", type=Path,
                   default=Path("analysis/initial_pca2_nn_20261008"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(protocol="initial_pca_nn_offline_v1", new_training=False,
                  source_bundle=str(args.bundle.resolve()),
                  leakage_rule="PCA and NN fit only on known-tool initial postures",
                  categories={})
    for category in CATEGORIES:
        blocks = read_category(args.bundle, category)
        known = heldout_pose_label_accuracy(blocks)
        unseen = leave_one_tool_out(blocks)
        report["categories"][category] = dict(
            heldout_pose_label=known, leave_one_tool_out=unseen,
            mean_known_label_accuracy={m:float(np.mean([r["label_accuracy"] for r in known[m]]))
                                       for m in METHODS},
            mean_cumulative_pca2_variance=float(np.mean([
                x["cumulative_pca2_variance"] for x in unseen])),
            mean_unseen_axis_cosine={m:float(np.mean([
                x["methods"][m]["mean_operation_axis_cosine"] for x in unseen]))
                for m in METHODS},
            mean_same_neighbour_as_raw13={m:float(np.mean([
                x["methods"][m]["same_neighbour_as_raw13_fraction"] for x in unseen]))
                for m in METHODS},
            mean_same_source_tool_as_raw13={m:float(np.mean([
                x["methods"][m]["same_source_tool_as_raw13_fraction"] for x in unseen]))
                for m in METHODS})
        print(category, "PCA2 variance", round(report["categories"][category][
            "mean_cumulative_pca2_variance"],3), "known label accuracy",
            report["categories"][category]["mean_known_label_accuracy"], flush=True)
    (args.output / "offline.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


if __name__ == "__main__":
    main()
