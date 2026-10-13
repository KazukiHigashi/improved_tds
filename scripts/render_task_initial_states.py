"""Render one frozen FF evaluation reset for each of the 15 tools.

The two views use the identical simulator state and Palm-facing camera. The
second view hides non-hand visual geoms; no dynamics step or policy action is
performed. Each chosen pose is the first FF test-pose ID for that tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from improved_tds.experiments.collect_posture_database import _actuated_q
from improved_tds.experiments.second_axis_feedback import reset_pristine_simulator
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments
from render_matched_representatives import PalmCamera
from simulate_initial_pose_conditioning import create_env


BUNDLE = ROOT / "analysis/paper15_paper_bundle_20260928"
TRAIN = ROOT / "runs/paper15_unified_3m_slew_20260927"
OUTPUT = ROOT / "analysis/paper_ff_fb_generalization_20261008/initial_states"
TASKS = tuple(f"{category}_lite_{i}"
              for category in ("scissors", "push", "spray") for i in range(1, 6))


def hand_only_frame(camera: PalmCamera) -> tuple[np.ndarray, int, int]:
    scene = camera.renderer.scene
    model = camera.model
    hand, hidden = 0, 0
    for index in range(scene.ngeom):
        geom = scene.geoms[index]
        name = None
        if (int(geom.objtype) == int(mujoco.mjtObj.mjOBJ_GEOM)
                and 0 <= int(geom.objid) < model.ngeom):
            body_id = int(model.geom_bodyid[int(geom.objid)])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name is not None and (name.startswith("rh_") or name.startswith("robot0:")):
            hand += 1
        else:
            geom.rgba[3] = 0.0
            hidden += 1
    if hand < 5 or hidden < 1:
        raise ValueError(f"unexpected hand-only scene: hand={hand}, hidden={hidden}")
    return camera.renderer.render().copy(), hand, hidden


def render_task(task: str, output: Path) -> dict:
    report_path = BUNDLE / task / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    test_poses = [int(x) for x in report["test_poses"]]
    if len(test_poses) != 8:
        raise ValueError(f"unexpected FF pose split: {task}")
    pose = test_poses[0]
    env, pool, key, lo, hi = create_env(task, BUNDLE, TRAIN, int(report["seed"]))
    camera = None
    try:
        if pool.pose_count != 32:
            raise ValueError(f"unexpected initial-pose pool: {task}")
        reset_pristine_simulator(env)
        env.reset(seed=int(report["seed"]) + 1000 * pose,
                  options={"initial_pose_index": pose, key: lo + .5 * (hi - lo)})
        initial_q = np.asarray(_actuated_q(env), dtype=float)
        camera = PalmCamera(env)
        full = camera.render()
        camera_dot = float(camera.last_alignment_dot)
        hand_only, hand_count, hidden_count = hand_only_frame(camera)
        folder = output / task
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(full).save(folder / "initial_state.jpg", quality=90, subsampling=0)
        Image.fromarray(hand_only).save(folder / "initial_hand.jpg", quality=90, subsampling=0)
        return dict(task=task, pose_id=pose, pool_size=int(pool.pose_count),
                    reset_seed=int(report["seed"]) + 1000 * pose,
                    nominal_goal=float(lo + .5 * (hi - lo)),
                    target_range=[float(lo), float(hi)],
                    initial_actuated_q=initial_q.tolist(),
                    camera_dot_palm_normal=camera_dot,
                    hand_scene_geoms=hand_count, hidden_scene_geoms=hidden_count,
                    state_image=f"{task}/initial_state.jpg",
                    hand_image=f"{task}/initial_hand.jpg",
                    report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest())
    finally:
        if camera is not None:
            camera.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    args = parser.parse_args()
    _register_tool_manip_gym_environments()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    existing = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file() else {"tasks": []})
    records = {item["task"]: item for item in existing["tasks"]}
    for task in args.tasks:
        records[task] = render_task(task, output)
        manifest = dict(protocol="frozen_ff_test_initial_state_palm_facing_v1",
                        no_simulation_step=True,
                        tasks=[records[name] for name in TASKS if name in records])
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False,
                                            indent=2), encoding="utf-8")
        print(f"rendered {task} pose {records[task]['pose_id']}", flush=True)


if __name__ == "__main__":
    main()
