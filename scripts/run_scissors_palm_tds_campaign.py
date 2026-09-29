"""Train under the palm-sphere criterion, collect successes, then run direct TDS.

This is a one-shot launcher: no external process polling is needed. Outputs are
kept separate from all pre-palm-sphere experiments.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--resume", action="store_true", help="resume the existing run after collection failure")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    run_root = args.run_root.resolve()
    analysis_root = args.analysis_root.resolve()
    campaign_root = run_root.parent / (run_root.name + "_campaign")
    if args.resume:
        if not run_root.is_dir() or not campaign_root.is_dir() or analysis_root.exists():
            raise ValueError("Resume requires existing training/campaign outputs and no TDS output")
    else:
        if any(path.exists() for path in (run_root, analysis_root, campaign_root)):
            raise FileExistsError("Run, analysis, or campaign output already exists")
        campaign_root.mkdir(parents=True)

    def stage(name, **extra):
        (campaign_root / "status.json").write_text(
            json.dumps(dict(stage=name, utc=datetime.now(timezone.utc).isoformat(),
                            run_root=str(run_root), analysis_root=str(analysis_root), **extra), indent=2),
            encoding="utf-8")

    def execute(name, command, env=None):
        (campaign_root / f"{name}_command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        with (campaign_root / f"{name}.log").open("w", encoding="utf-8") as stdout, \
                (campaign_root / f"{name}.err.log").open("w", encoding="utf-8") as stderr:
            subprocess.run(command, cwd=project, env=env or os.environ.copy(), stdout=stdout, stderr=stderr,
                           check=True)

    try:
        if args.resume:
            stage("resuming_success_collection")
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([str(project / "src"), *(
                str(project.parent / "tool-manip-gym" / name)
                for name in ("gym-scissors", "gym-push", "gym-spray", "gym-slider"))])
            task = run_root / "scissors_lite_5"
            execute("collection_resume", [sys.executable, "-m",
                    "improved_tds.experiments.collect_reach_successes", "--env-id", "ScissorsVariant5-v0",
                    "--model", str(task / "best_success_policy.zip"),
                    "--pose-pool-dir", str(run_root / "pose_pool"),
                    "--output", str(task / "success_postures.npz"), "--per-bin", "20", "--seed", "9305",
                    "--scissors-control-version", "smooth"], env=env)
            status_doc = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
            status_doc["tasks"]["scissors_lite_5"] = "completed"
            (run_root / "status.json").write_text(json.dumps(status_doc, indent=2), encoding="utf-8")
            (task / "phase.json").write_text(json.dumps(dict(phase="completed",
                utc=datetime.now(timezone.utc).isoformat())), encoding="utf-8")
        else:
            stage("training_and_collecting")
            execute("training", [sys.executable, str(project / "scripts/run_scissors_retention_v4.py"),
                                 "--output", str(run_root), "--timesteps", str(args.timesteps),
                                 "--variants", "3 4 5", "--n-envs", "4", "--version", "smooth",
                                 "--collect-successes"])
        status = json.loads((run_root / "status.json").read_text(encoding="utf-8"))["tasks"]
        if any(value != "completed" for value in status.values()):
            raise RuntimeError(f"Training or 200-posture collection incomplete: {status}")
        for variant in (3, 4, 5):
            task = f"scissors_lite_{variant}"
            with np.load(run_root / task / "success_postures.npz", allow_pickle=False) as data:
                meta = json.loads(data["metadata_json"].item())
                if (not meta.get("complete") or len(data["q"]) != 200 or not data["success"].all()
                        or meta["contract"].get("grasp_criterion") !=
                        "hinge_center_inside_palm_attached_original_sphere"):
                    raise RuntimeError(f"{task}: invalid new-criterion success database")
        stage("tds_evaluation")
        execute("tds", [sys.executable, str(project / "scripts/run_reach_tds_direct.py"),
                        "--source-root", str(run_root), "--output", str(analysis_root),
                        "--workers", "4"])
        status = json.loads((analysis_root / "status.json").read_text(encoding="utf-8"))["jobs"]
        if any(value != "completed" for value in status.values()):
            raise RuntimeError(f"TDS evaluation incomplete: {status}")
        stage("completed", report=str(analysis_root / "index.html"))
    except Exception as exc:
        stage("failed", error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
