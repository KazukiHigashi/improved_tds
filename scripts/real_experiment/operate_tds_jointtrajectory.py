#!/usr/bin/env python3
"""Plan or execute one open-loop TDS Shadow Hand trajectory.

The command transport follows the original real_experiment ROS1
FollowJointTrajectory action.  No ROS or serial connection is made unless
--execute is supplied.  A reviewed joint-map and plan digest are mandatory.
The simulation-to-robot transform and hardware limits are deliberately NOT
guessed from the MuJoCo model.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


ACTION_NAME = "/rh_trajectory_controller/follow_joint_trajectory"


def finite_vector(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return result


def inverse_score(calibration: Path, angle: float) -> float:
    """Invert independently measured real calibration; never extrapolate."""
    import csv

    with calibration.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    points = [(float(r["score"]), float(r["angle_rad"])) for r in rows
              if r.get("valid", "1").strip().lower() in ("1", "true", "yes")]
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("real calibration requires at least three finite valid points")
    groups: dict[float, list[float]] = {}
    for score, measured in points:
        groups.setdefault(round(score, 8), []).append(measured)
    if len(groups) < 3:
        raise ValueError("real calibration requires three distinct valid scores")
    scores = np.asarray(sorted(groups))
    angles = np.asarray([float(np.median(groups[score])) for score in scores])
    if not (np.diff(scores) > 0).all() or not (np.diff(angles) > 0).all():
        raise ValueError("real calibration must be strictly monotone; select a valid branch")
    if not angles[0] <= angle <= angles[-1]:
        raise ValueError("target angle is outside the measured real inverse domain")
    return float(np.interp(angle, angles, scores))


def make_plan(report: dict, joint_map: dict, score: float, seconds: float,
              sample_hz: float, target_angle: float | None) -> dict:
    config = report["configurations"][joint_map["configuration"]]
    selection = config.get("selection")
    if not selection:
        raise ValueError("report has no valid monotone calibration interval")
    lo, hi = sorted((float(selection["rho_start"]), float(selection["rho_end"])))
    if not lo <= score <= hi:
        raise ValueError(f"score {score} is outside selected interval [{lo}, {hi}]")
    mean = finite_vector(report["mean_q"], "mean_q")
    axis = finite_vector(config["fit"]["operation_axis"], "operation_axis")
    joints = joint_map["joints"]
    if len(mean) != len(axis) or len(joints) != len(mean):
        raise ValueError("simulation vector and reviewed joint map differ in length")
    if len({j["name"] for j in joints}) != len(joints):
        raise ValueError("robot joint names must be unique")
    if config["dof"] != 1:
        raise ValueError("this one-DoF script requires a *_1d configuration")
    if seconds < 5 or sample_hz < 2 or not math.isfinite(seconds * sample_hz):
        raise ValueError("trajectory must be at least five seconds and >= 2 Hz")
    q_sim = mean + float(score) * float(config["operation_scale"]) * axis
    goal = np.empty(len(joints))
    for index, joint in enumerate(joints):
        if not joint.get("verified", False):
            raise ValueError(f"joint mapping {index} was not independently verified")
        sign, offset = float(joint["sign"]), float(joint["offset_rad"])
        sim_lower, sim_upper = float(joint["sim_min_rad"]), float(joint["sim_max_rad"])
        lower, upper = float(joint["min_rad"]), float(joint["max_rad"])
        velocity = float(joint["max_velocity_rad_s"])
        if (sign not in (-1., 1.) or not sim_lower < sim_upper
                or not lower < upper or velocity <= 0):
            raise ValueError(f"invalid mapping, limit, or velocity for {joint['name']}")
        q_sim[index] = np.clip(q_sim[index], sim_lower, sim_upper)
        goal[index] = sign * q_sim[index] + offset
        if not lower <= goal[index] <= upper:
            raise ValueError(f"goal exceeds reviewed limit of {joint['name']}")
    if not np.isfinite(goal).all():
        raise ValueError("nonfinite robot goal")
    plan = {
        "schema": "tds_shadow_hand_trajectory_v1",
        "task": report["task"], "configuration": joint_map["configuration"],
        "source_report_sha256": hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest(),
        "score": float(score), "target_angle_rad": target_angle,
        "duration_s": seconds, "hold_s": 2.0, "sample_hz": sample_hz,
        "joint_names": [j["name"] for j in joints],
        "expected_start_rad": finite_vector(joint_map["expected_start_rad"],
                                            "expected_start_rad").tolist(),
        "goal_rad": goal.tolist(),
        "clipped_sim_goal_rad": q_sim.tolist(),
        "limits": [[float(j["min_rad"]), float(j["max_rad"])] for j in joints],
        "max_velocity_rad_s": [float(j["max_velocity_rad_s"]) for j in joints],
        "action_name": ACTION_NAME,
    }
    if len(plan["expected_start_rad"]) != len(joints):
        raise ValueError("expected_start_rad must contain every mapped joint")
    plan["sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def execute(plan: dict, approval: str, start_tolerance: float,
            serial_port: str | None, serial_baud: int) -> float:
    if plan["sha256"] != approval:
        raise ValueError("--approve-sha must equal the displayed plan SHA-256")
    # These imports are intentionally deferred so offline review needs no ROS.
    import rospy
    import actionlib
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

    rospy.init_node("operate_tds_jointtrajectory", anonymous=True)
    state = rospy.wait_for_message("/joint_states", JointState, timeout=10.0)
    if state.header.stamp.to_sec() <= 0 or (rospy.Time.now() - state.header.stamp).to_sec() > 1.0:
        raise RuntimeError("joint-state feedback is unstamped or stale")
    positions = dict(zip(state.name, state.position))
    names = plan["joint_names"]
    if not all(n in positions for n in names):
        raise RuntimeError("/joint_states does not contain all mapped joints")
    start = np.asarray([positions[n] for n in names], dtype=float)
    goal = np.asarray(plan["goal_rad"], dtype=float)
    limits = np.asarray(plan["limits"], dtype=float)
    velocities = np.asarray(plan["max_velocity_rad_s"], dtype=float)
    if not np.isfinite(start).all() or np.any(start < limits[:, 0]) or np.any(start > limits[:, 1]):
        raise RuntimeError("live starting joint state is invalid or outside reviewed limits")
    duration = float(plan["duration_s"])
    if np.any(abs(goal - start) / duration > velocities):
        raise RuntimeError("5-second move exceeds a reviewed per-joint speed limit")
    expected_start = np.asarray(plan["expected_start_rad"], dtype=float)
    if np.any(abs(expected_start - start) > start_tolerance):
        raise RuntimeError("robot is not in the manually approved starting pose")
    client = actionlib.SimpleActionClient(ACTION_NAME, FollowJointTrajectoryAction)
    if not client.wait_for_server(rospy.Duration(10.0)):
        raise RuntimeError("Shadow Hand trajectory action is unavailable")
    trajectory = JointTrajectory()
    trajectory.joint_names = names
    count = max(2, math.ceil(duration * float(plan["sample_hz"])))
    for step in range(1, count + 1):
        fraction = step / count
        point = JointTrajectoryPoint()
        point.positions = (start + fraction * (goal - start)).tolist()
        point.time_from_start = rospy.Duration.from_sec(duration * fraction)
        trajectory.points.append(point)
    trajectory.header.stamp = rospy.Time.now() + rospy.Duration.from_sec(0.2)
    action = FollowJointTrajectoryGoal()
    action.trajectory = trajectory
    reader = None
    if serial_port:
        import serial
        reader = serial.Serial(serial_port, serial_baud, timeout=1)
    try:
        client.send_goal(action)
        if not client.wait_for_result(rospy.Duration.from_sec(duration + 5.0)):
            client.cancel_goal()
            raise RuntimeError("trajectory timed out; goal cancelled")
        result = client.get_result()
        if result is None or getattr(result, "error_code", 0) != 0:
            raise RuntimeError(f"trajectory controller reported failure: {result}")
        rospy.sleep(float(plan["hold_s"]))
        if reader:
            reader.reset_input_buffer()  # Discard pre-motion/stale samples.
            samples = []
            for _ in range(5):
                line = reader.readline().decode("ascii", errors="strict").strip()
                samples.append(math.radians(float(line)))
            angle = float(np.median(samples))
            if not math.isfinite(angle):
                raise RuntimeError("angle sensor returned nonfinite data")
            print(json.dumps({"score": plan["score"], "angle_rad": angle,
                              "samples_rad": samples, "valid": True}))
            return angle
        raise RuntimeError("angle sensor was not configured")
    except BaseException:
        client.cancel_goal()
        raise
    finally:
        if reader:
            reader.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--joint-map", required=True, type=Path,
                        help="independently reviewed JSON with configuration and 13 ordered joints")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--score", type=float, help="calibration score")
    choice.add_argument("--target-angle", type=float, help="target angle in radians")
    parser.add_argument("--real-calibration", type=Path,
                        help="CSV with score,angle_rad,valid; required for target control")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--sample-hz", type=float, default=20.0)
    parser.add_argument("--execute", action="store_true", help="actually send one ROS action")
    parser.add_argument("--approve-sha", default="", help="SHA-256 printed by dry-run")
    parser.add_argument("--start-tolerance", type=float, default=0.1)
    parser.add_argument("--serial-port", help="Arduino tool-angle sensor, degrees per line")
    parser.add_argument("--serial-baud", type=int, default=115200)
    parser.add_argument("--log-csv", type=Path, help="append measured live calibration/evaluation row")
    args = parser.parse_args(argv)
    if not 0 < args.start_tolerance <= 0.1 or not math.isfinite(args.start_tolerance):
        parser.error("--start-tolerance must be finite and in (0, 0.1] rad")
    if args.execute and (not args.serial_port or args.log_csv is None):
        parser.error("live experiments require --serial-port and --log-csv")
    if args.execute and not args.log_csv.parent.is_dir():
        parser.error("log parent directory must already exist")
    if args.score is not None:
        score = args.score
    else:
        if args.real_calibration is None:
            parser.error("--target-angle requires --real-calibration")
        score = inverse_score(args.real_calibration, args.target_angle)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    mapping = json.loads(args.joint_map.read_text(encoding="utf-8"))
    plan = make_plan(report, mapping, score, args.duration, args.sample_hz, args.target_angle)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    print("DRY RUN" if not args.execute else "EXECUTE REQUESTED", file=sys.stderr)
    if args.execute:
        if plan["sha256"] != args.approve_sha:
            parser.error("--approve-sha must exactly match the dry-run plan digest")
        path = args.log_csv
        existing = path.exists()
        with path.open("a", newline="", encoding="utf-8") as output:
            measured = execute(plan, args.approve_sha, args.start_tolerance,
                               args.serial_port, args.serial_baud)
            writer = csv.DictWriter(output, fieldnames=("utc", "task", "configuration",
                "score", "angle_rad", "target_angle_rad", "plan_sha256", "valid"))
            if not existing:
                writer.writeheader()
            writer.writerow({"utc": datetime.now(timezone.utc).isoformat(),
                "task": plan["task"], "configuration": plan["configuration"],
                "score": plan["score"], "angle_rad": measured,
                "target_angle_rad": plan["target_angle_rad"],
                "plan_sha256": plan["sha256"], "valid": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
