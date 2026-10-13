"""Replay ranked matched-nine trials and save Palm-facing frames every 25 steps.

The rank is computed from every complete, finite evaluation trial with a
nonzero target in each method/arm across all eligible tools. It does not change the frozen
experiment JSON, calibration, or evaluation results; it only adds RGB capture.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from improved_tds.control.axis_feedback import AxisFeedbackConfig
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds
from improved_tds.experiments.second_axis_feedback import (
    reset_pristine_simulator, run_trial, inverse_score as fb_inverse_score)
from improved_tds.experiments.simple_synergy import direct_trajectory, simulation_step_seconds
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from evaluate_initial_pose_conditioning import fit_operation, read_category
from simulate_initial_pose_conditioning import create_env, fit_angle_inverse, fit_local
from replay_matched_nine_ff import frozen_score
from replay_matched_nine_generalization import command_score


RESULT = ROOT / "analysis/matched_nine_goals_clipped_20261008"
BUNDLE = ROOT / "analysis/paper15_paper_bundle_20260928"
TRAIN = ROOT / "runs/paper15_unified_3m_slew_20260927"
OUT = ROOT / "analysis/paper_ff_fb_generalization_20261008/representative_frames_nonzero"
TASKS = tuple(f"{cat}_lite_{i}" for cat in ("scissors", "push", "spray") for i in range(1, 6))
FB_TASKS = ("scissors_lite_3", "scissors_lite_4", "push_lite_3", "spray_lite_3")
UNKNOWN_ARMS = ("common_zero", "pca2_nn_zero", "common_three_probe",
                "selected_three_probe", "local_pls_oracle", "local_pls_three_probe_oracle")
GROUPS = (tuple(f"ff_{method}" for method in ("pca", "pls"))
          + tuple(f"unknown_{arm}" for arm in UNKNOWN_ARMS)
          + tuple(f"fb_{method}_{condition}" for method in ("pca", "pls")
                  for condition in ("axis1_fb", "axis2_fb")))
RANKS = ("best", "median", "worst")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nonzero_target(value: float) -> bool:
    return not math.isclose(float(value), 0.0, rel_tol=0.0, abs_tol=1e-12)


def candidates(group: str) -> list[dict]:
    items = []
    if group.startswith("ff_"):
        method = group[3:]
        for task in TASKS:
            result = read(RESULT / "ff" / f"{task}.json")
            span = result["target_range"][1] - result["target_range"][0]
            for row in result["rows"]:
                if (row["method"] == method and nonzero_target(row["target_angle"])
                        and row["finite"]
                        and not row["terminated"] and not row["truncated"]):
                    items.append(dict(task=task, pose=int(row["pose"]),
                        target=float(row["target_angle"]), fraction=float(row["target_fraction"]),
                        achieved=float(row["achieved_angle"]), error_percent=
                        100 * abs(row["achieved_angle"]-row["target_angle"])/span,
                        held=bool(row["held"]), group=group))
    elif group.startswith("unknown_"):
        arm = group[len("unknown_"):]
        for task in TASKS:
            result = read(RESULT / "generalization" / f"{task}.json")
            span = result["target_range"][1] - result["target_range"][0]
            for row in result["rows"]:
                if (row["arm"] == arm and nonzero_target(row["target_angle"])
                        and row["finite"]
                        and not row["terminated"] and not row["truncated"]):
                    items.append(dict(task=task, pose=int(row["pose"]),
                        target=float(row["target_angle"]), fraction=float(row["target_fraction"]),
                        achieved=float(row["achieved_angle"]), error_percent=
                        100 * abs(row["achieved_angle"]-row["target_angle"])/span,
                        held=bool(row["held"]), group=group,
                        selected_source=row.get("selected_source")))
    else:
        _, method, condition = group.split("_", 2)
        for task in FB_TASKS:
            result = read(RESULT / "fb" / task / "report.json")
            span = result["target_range"][1] - result["target_range"][0]
            for row in result["methods"][method]["trials"]:
                if (row["condition"] == condition and nonzero_target(row["target"])
                        and row["complete"]
                        and not row["numerical_failure"] and math.isfinite(row["achieved"])):
                    items.append(dict(task=task, pose=int(row["pose"]),
                        target=float(row["target"]), fraction=float(
                            (row["target"]-result["target_range"][0])/span),
                        achieved=float(row["achieved"]), error_percent=
                        100 * abs(row["achieved"]-row["target"])/span,
                        held=bool(row["valid"]), group=group))
    if not items:
        raise ValueError(f"no complete finite cases: {group}")
    return sorted(items, key=lambda x: (x["error_percent"], x["task"],
                                        x["pose"], x["target"]))


def select_cases(groups: tuple[str, ...] = GROUPS, ranks: tuple[str, ...] = RANKS) -> list[dict]:
    chosen = []
    for group in groups:
        pool = candidates(group)
        for rank in ranks:
            index = {"best": 0, "median": (len(pool)-1)//2, "worst": len(pool)-1}[rank]
            chosen.append(dict(pool[index], rank=rank, rank_index=index+1, pool_count=len(pool)))
    return chosen


class PalmCamera:
    """Look from the palmar side: view direction dot palm normal is -1."""

    def __init__(self, env):
        raw = env.unwrapped
        self.model, self.data = raw.model, raw.data
        self.palm = next((mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                          for name in ("rh_palm", "robot0:palm")
                          if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0), -1)
        if self.palm < 0:
            raise ValueError("Palm body not found")
        self.renderer = mujoco.Renderer(self.model, width=512, height=384)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.last_alignment_dot = None

    def render(self) -> np.ndarray:
        rotation = self.data.xmat[self.palm].reshape(3, 3)
        # The palm collision boxes lie in the local x-z plane; grasp_site is
        # on local -y. The palmar surface normal is therefore local -y.
        normal = rotation @ np.array([0., -1., 0.])
        lookat = self.data.xpos[self.palm] + rotation @ np.array([0., -.01, .06])
        self.camera.lookat[:] = lookat
        self.camera.distance = .36
        # MuJoCo's free-camera azimuth places the camera on the opposite side
        # from this world-space offset angle. Add 180 deg, then VERIFY the
        # actual rendered camera position instead of trusting angle algebra.
        self.camera.azimuth = float(np.degrees(np.arctan2(normal[1], normal[0])) + 180.)
        # MuJoCo also inverts the vertical component of a free camera offset.
        self.camera.elevation = -float(np.degrees(np.arcsin(np.clip(normal[2], -1., 1.))))
        self.renderer.update_scene(self.data, camera=self.camera)
        cameras = self.renderer.scene.camera
        camera_position = .5 * (np.asarray(cameras[0].pos) + np.asarray(cameras[1].pos))
        view_direction = lookat - camera_position
        view_direction /= np.linalg.norm(view_direction)
        self.last_alignment_dot = float(np.dot(view_direction, normal))
        if self.last_alignment_dot > -.999:
            raise RuntimeError(f"camera is not facing Palm: dot={self.last_alignment_dot:.6f}")
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


def ff_goal(case: dict, report: dict, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    config = report["configurations"][case["group"][3:] + "_1d"]
    rho, _ = frozen_score(config, case["target"])
    requested = (np.asarray(report["mean_q"], float) + rho
        * float(config["operation_scale"]) * np.asarray(config["fit"]["operation_axis"], float))
    return np.clip(requested, low, high)


def unknown_goal(case: dict, task: str, report: dict, result: dict,
                 low: np.ndarray, high: np.ndarray) -> np.ndarray:
    category, _, number = task.partition("_lite_")
    blocks = read_category(BUNDLE, category)
    held = blocks[int(number)-1]
    known = [block for block in blocks if block["name"] != task]
    center, axis = fit_operation(known)
    common_inverse, common_bounds = fit_angle_inverse(known, center, axis)
    common = (center, axis, common_inverse, common_bounds)
    arm = case["group"][len("unknown_"):]
    if arm in ("common_zero", "common_three_probe"):
        package = common
        response = (result["candidate_models"]["common"]["response"]
                    if arm == "common_three_probe" else None)
    elif arm == "pca2_nn_zero":
        package = fit_local(next(block for block in known
                                 if block["name"] == case["selected_source"]))
        response = None
    elif arm == "selected_three_probe":
        selected = result["selected_candidate"]
        package = common if selected in (None, "common") else fit_local(next(
            block for block in known if block["name"] == selected))
        response = (result["candidate_models"][selected]["response"]
                    if selected is not None else None)
    else:
        package = fit_local(held)
        response = result["oracle_response"] if arm == "local_pls_three_probe_oracle" else None
    score, _ = command_score(case["fraction"], case["target"], package, response)
    return np.clip(package[0] + score * package[1], low, high)


def capture_fb_without_changing_controller(env, capture, run):
    """Observe reset/step outputs without editing the frozen FB controller."""
    original_reset, original_step = env.reset, env.step
    steps = 0

    def observed_reset(*args, **kwargs):
        nonlocal steps
        observation, info = original_reset(*args, **kwargs)
        steps = 0
        capture(0, observation, info)
        return observation, info

    def observed_step(action):
        nonlocal steps
        output = original_step(action)
        steps += 1
        observation, _, terminated, truncated, info = output
        if steps % 25 == 0 or terminated or truncated:
            capture(steps, observation, info)
        return output

    env.reset, env.step = observed_reset, observed_step
    try:
        return run()
    finally:
        env.reset, env.step = original_reset, original_step


def render_case(case: dict, output: Path) -> dict:
    task, group, rank = case["task"], case["group"], case["rank"]
    folder = output / group / rank
    folder.mkdir(parents=True, exist_ok=True)
    source = read(BUNDLE / task / "report.json")
    fb = group.startswith("fb_")
    fb_result = read(RESULT / "fb" / task / "report.json") if fb else None
    env, _, key, lo, hi = create_env(task, BUNDLE, TRAIN, source["seed"])
    camera = PalmCamera(env)
    frames = []
    dt = simulation_step_seconds(env)

    def capture(step, obs, info):
        image = Image.fromarray(camera.render())
        name = f"step_{step:04d}.jpg"
        image.save(folder / name, quality=85, subsampling=0)
        frames.append(dict(step=int(step), time_seconds=float(step*dt),
            image=f"{group}/{rank}/{name}", tool=float(obs["achieved_goal"][0]),
            camera_dot_palm_normal=float(camera.last_alignment_dot),
            on_palm=bool(info.get("is_on_palm")) if "is_on_palm" in info else None,
            dropped=bool(info.get("object_dropped", info.get("bottle_dropped", False))),
            tipped=bool(info.get("object_tipped", info.get("bottle_tipped", False)))))
    try:
        reset_pristine_simulator(env)
        if fb:
            _, method, condition = group.split("_", 2)
            config = source["configurations"][method+"_1d"]
            axes = [np.asarray(config["fit"]["operation_axis"], float)*config["operation_scale"],
                    np.asarray(config["fit"]["grasp_axis"], float)*config["grasp_scale"]]
            domain = config["inverse_target_range"]
            rho = fb_inverse_score(config, float(np.clip(case["target"], *domain)))
            calibration = next(row["sensitivity"] for row in fb_result["methods"][method]["calibration"]
                               if row["condition"] == condition
                               and np.isclose(row["target"], case["target"], atol=1e-9))
            settings = dict(tool_span=hi-lo,
                            feedback=AxisFeedbackConfig(**fb_result["feedback_settings"]))
            result = capture_fb_without_changing_controller(env, capture, lambda: run_trial(
                env, key, case["pose"], source["seed"]+1000*case["pose"],
                case["target"], np.asarray(source["mean_q"], float), axes, rho,
                condition, calibration, settings,
                "task" if task.startswith("scissors") else "no_fall", task.startswith("spray")))
            achieved, held = float(result["achieved"]), bool(result["valid"])
        else:
            low, high, _ = _control_bounds(env)
            if group.startswith("ff_"):
                goal = ff_goal(case, source, low, high)
            else:
                trial = read(RESULT / "generalization" / f"{task}.json")
                goal = unknown_goal(case, task, source, trial, low, high)
            result = direct_trajectory(env, case["pose"], key, case["target"],
                source["seed"]+1000*case["pose"], [goal],
                support_criterion="task" if task.startswith("scissors") else "no_fall",
                require_floor_contact=task.startswith("spray"), capture=capture,
                capture_every=25, capture_initial=True)["outcomes"][0]
            achieved, held = float(result["achieved"]), bool(result["valid"])
    finally:
        camera.close()
        env.close()
    tolerance = 1e-4 if task.startswith("push") else 5e-3
    difference = abs(achieved-case["achieved"])
    if difference > tolerance or held != case["held"]:
        raise RuntimeError(f"replay mismatch {group}/{rank}/{task}: "
                           f"achieved {achieved}/{case['achieved']} "
                           f"held {held}/{case['held']}")
    return dict(case, replay_achieved=achieved, replay_held=held,
                replay_abs_difference=difference, step_seconds=dt,
                camera_local_normal=[0., -1., 0.],
                view_direction_dot_palm_normal=float(max(
                    frame["camera_dot_palm_normal"] for frame in frames)), frames=frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS)
    parser.add_argument("--ranks", nargs="+", choices=RANKS, default=RANKS)
    parser.add_argument("--force", action="store_true", help="replace selected generated frames")
    args = parser.parse_args()
    _register_tool_manip_gym_environments()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    requested = select_cases(tuple(args.groups), tuple(args.ranks))
    existing = read(output / "manifest.json") if (output / "manifest.json").is_file() else {}
    records = {(row["group"], row["rank"]): row for row in existing.get("cases", [])}
    for index, case in enumerate(requested, 1):
        key = (case["group"], case["rank"])
        old = records.get(key)
        same_case = (old is not None and all(old.get(field) == case[field]
                    for field in ("task", "pose", "target", "achieved", "held",
                                  "error_percent", "rank_index", "pool_count")))
        if (same_case and not args.force
                and all((output / frame["image"]).is_file() for frame in old["frames"])):
            print(f"{index}/{len(requested)} existing {case['group']}/{case['rank']}", flush=True)
            continue
        print(f"{index}/{len(requested)} render {case['group']}/{case['rank']} "
              f"{case['task']} pose={case['pose']}", flush=True)
        records[key] = render_case(case, output)
        manifest = dict(protocol="matched_nine_ranked_representative_frames_nonzero_v1",
            selection="nonzero target and complete finite terminal normalized absolute error; rank within arm across all eligible tools",
            rank_median="lower middle element in sorted pool", interval_steps=25,
            camera="local palm -y normal; camera view direction dot normal = -1",
            cases=[records[key] for key in sorted(records)])
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
