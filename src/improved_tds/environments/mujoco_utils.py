from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np


@dataclass(frozen=True)
class JointView:
    qpos_adr: int
    qpos_size: int
    qvel_adr: int
    qvel_size: int


def name2id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise KeyError(f"{name!r} is not present in the MuJoCo model")
    return obj_id


def id2name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    if name is None:
        raise KeyError(f"MuJoCo object id {obj_id} has no name")
    return name


def joint_view(model: mujoco.MjModel, joint_name: str) -> JointView:
    joint_id = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    joint_type = model.jnt_type[joint_id]
    qpos_adr = int(model.jnt_qposadr[joint_id])
    qvel_adr = int(model.jnt_dofadr[joint_id])

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return JointView(qpos_adr, 7, qvel_adr, 6)
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return JointView(qpos_adr, 4, qvel_adr, 3)
    if joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        return JointView(qpos_adr, 1, qvel_adr, 1)
    raise ValueError(f"Unsupported MuJoCo joint type {joint_type} for {joint_name}")


def get_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> np.ndarray:
    view = joint_view(model, joint_name)
    return data.qpos[view.qpos_adr : view.qpos_adr + view.qpos_size].copy()


def get_joint_qvel(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> np.ndarray:
    view = joint_view(model, joint_name)
    return data.qvel[view.qvel_adr : view.qvel_adr + view.qvel_size].copy()


def set_joint_qpos(
    model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float | np.ndarray
) -> None:
    view = joint_view(model, joint_name)
    value_array = np.asarray(value, dtype=np.float64).reshape(-1)
    if value_array.size != view.qpos_size:
        raise ValueError(
            f"{joint_name} expects {view.qpos_size} qpos values, got {value_array.size}"
        )
    data.qpos[view.qpos_adr : view.qpos_adr + view.qpos_size] = value_array


def set_joint_qvel(
    model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float | np.ndarray
) -> None:
    view = joint_view(model, joint_name)
    value_array = np.asarray(value, dtype=np.float64).reshape(-1)
    if value_array.size != view.qvel_size:
        raise ValueError(
            f"{joint_name} expects {view.qvel_size} qvel values, got {value_array.size}"
        )
    data.qvel[view.qvel_adr : view.qvel_adr + view.qvel_size] = value_array


@lru_cache(maxsize=32)
def robot_joint_names(model_ptr: int, names: tuple[str, ...]) -> tuple[str, ...]:
    del model_ptr
    return tuple(name for name in names if name.startswith("robot0:"))


def all_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        name
        for joint_id in range(model.njnt)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
    )


def robot_get_obs(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    names = robot_joint_names(id(model), all_joint_names(model))
    qpos = [get_joint_qpos(model, data, name).ravel() for name in names]
    qvel = [get_joint_qvel(model, data, name).ravel() for name in names]
    return (
        np.concatenate(qpos).astype(np.float32) if qpos else np.zeros(0, dtype=np.float32),
        np.concatenate(qvel).astype(np.float32) if qvel else np.zeros(0, dtype=np.float32),
    )


def actuator_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        name
        for actuator_id in range(model.nu)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id))
    )

