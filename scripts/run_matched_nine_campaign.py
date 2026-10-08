"""Run/resume the 15-task matched-nine evaluation without retraining.

Each child gets an individual log. Completed files are validated and skipped;
the launcher never deletes or overwrites earlier results. Inspect status.json
after the process exits rather than polling it continuously.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = tuple(f"{category}_lite_{i}" for category in ("scissors", "push", "spray")
              for i in range(1, 6))
FB_TASKS = ("scissors_lite_3", "scissors_lite_4", "push_lite_3", "spray_lite_3")
CODE_FILES = (ROOT / "scripts/replay_matched_nine_ff.py",
              ROOT / "scripts/replay_matched_nine_generalization.py",
              ROOT / "src/improved_tds/experiments/second_axis_feedback.py")


def code_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in CODE_FILES}


def valid_result(root: Path, stage: str, task: str) -> bool:
    path = (root / stage / task / "report.json" if stage == "fb" else
            root / stage / f"{task}.json")
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("task") != task:
            return False
        if stage == "ff":
            return len(data["rows"]) == 2 * 8 * 9
        if stage == "generalization":
            return len(data["rows"]) == 6 * 8 * 9
        return all(len(data["methods"][m]["trials"]) == 3 * 8 * 9
                   for m in ("pca", "pls"))
    except (KeyError, ValueError, TypeError, OSError):
        return False


def run_one(root: Path, stage: str, task: str) -> dict:
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if valid_result(root, stage, task):
        return dict(stage=stage, task=task, status="already_complete")
    if stage == "ff":
        cmd = [sys.executable, str(ROOT / "scripts/replay_matched_nine_ff.py"),
               "--task", task, "--output", str(root / "ff")]
    elif stage == "generalization":
        cmd = [sys.executable, str(ROOT / "scripts/replay_matched_nine_generalization.py"),
               "--task", task, "--output", str(root / "generalization")]
    else:
        target = root / "fb" / task
        if target.exists():
            # The underlying FB runner intentionally refuses to overwrite any
            # partial result. Retain it for diagnosis rather than deleting it.
            return dict(stage=stage, task=task, status="partial_output_needs_review")
        cmd = [sys.executable, "-m", "improved_tds.experiments.second_axis_feedback",
               "--task", task, "--bundle",
               str(ROOT / "analysis/paper15_paper_bundle_20260928"),
               "--output", str(target), "--calibration-poses", "8",
               "--test-poses", "8", "--targets", "9", "--full-rl-grid"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(ROOT / "src"), str(ROOT / "scripts"), env.get("PYTHONPATH", ""))))
    log = log_dir / f"{stage}_{task}.log"
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, check=False)
    return dict(stage=stage, task=task,
                status=("complete" if completed.returncode == 0
                        and valid_result(root, stage, task) else "failed"),
                exit_code=completed.returncode, log=str(log.relative_to(root)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=ROOT / "analysis/matched_nine_goals_20261008")
    parser.add_argument("--stage", choices=("all", "ff", "generalization", "fb"),
                        default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be 1–4")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    frozen_code = code_hashes()
    stages = ("ff", "generalization", "fb") if args.stage == "all" else (args.stage,)
    jobs = [(stage, task) for stage in stages
            for task in (FB_TASKS if stage == "fb" else TASKS)]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, root, stage, task):(stage, task)
                   for stage, task in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(result["stage"], result["task"], result["status"], flush=True)
            (root / "status.json").write_text(json.dumps(dict(
                protocol="common_full_rl_nine_goal_v1", no_retraining=True,
                code_sha256=frozen_code,
                updated_utc=datetime.now(timezone.utc).isoformat(),
                expected_jobs=len(jobs), completed_jobs=len(results),
                jobs=sorted(results, key=lambda r:(r["stage"],r["task"]))),
                ensure_ascii=False, indent=2), encoding="utf-8")
    if code_hashes() != frozen_code:
        raise SystemExit("Experiment source changed while jobs were running")
    if any(r["status"] not in ("complete", "already_complete") for r in results):
        raise SystemExit("One or more jobs failed; inspect status.json and per-task logs")


if __name__ == "__main__":
    main()
