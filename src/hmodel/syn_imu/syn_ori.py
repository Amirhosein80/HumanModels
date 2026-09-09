"""Synthetic orientation synthesis: gyro integration."""

from __future__ import annotations

import roma
import torch

__all__ = [
    "integrate_ori_gyro",
    "add_ori_noise",
    "unmount_to_bone",
]


def integrate_ori_gyro(
    rotation_imu: torch.Tensor,
    gyr: torch.Tensor,
    fps: int,
) -> torch.Tensor:
    """Integrate sensor orientation from gyroscope readings (GlobalPose ``skip_ESKF`` replica).

    Sequential matrix-product scan: ``R[0] = rotation_imu[0]``,
    ``R[t] = R[t-1] @ rotvec(gyr[t] / fps)``.

    Args:
        rotation_imu: (T, S, 3, 3) sensor orientations in world frame.
        gyr:          (T, S, 3) sensor-local angular velocity in rad/s.
        fps:          Sampling rate in Hz.

    Returns:
        rot_sensor_world: (T, S, 3, 3) integrated sensor orientations.
    """
    T, S = gyr.shape[0], gyr.shape[1]
    dtype, device = gyr.dtype, gyr.device
    rot_sensor_world = torch.empty(T, S, 3, 3, dtype=dtype, device=device)
    rot_sensor_world[0] = rotation_imu[0]
    dR_gyr = roma.rotvec_to_rotmat((gyr / fps).reshape(T * S, 3)).reshape(T, S, 3, 3)
    # TODO: vectorize — this is a sequential matrix-product scan
    for step in range(1, T):
        rot_sensor_world[step] = rot_sensor_world[step - 1] @ dR_gyr[step]
    return rot_sensor_world


def add_ori_noise(rot_sensor_world: torch.Tensor, k: float) -> torch.Tensor:
    """Apply post-integration orientation noise (GlobalPose ``nR`` replica).

    Right-multiplies each orientation with ``rotvec(randn * 0.1 * k)``.

    Args:
        rot_sensor_world: (T, S, 3, 3) sensor orientations in world frame.
        k:                Noise constant ``sqrt(pi / 8)``.

    Returns:
        Noisy orientations, same shape.
    """
    T, S = rot_sensor_world.shape[0], rot_sensor_world.shape[1]
    dtype, device = rot_sensor_world.dtype, rot_sensor_world.device
    nR = roma.rotvec_to_rotmat(
        (torch.randn(T, S, 3, dtype=dtype, device=device) * 0.1 * k).reshape(T * S, 3)
    ).reshape(T, S, 3, 3)
    return rot_sensor_world @ nR


def unmount_to_bone(rot_sensor_world: torch.Tensor, rbs: torch.Tensor) -> torch.Tensor:
    """Unmount sensor orientations to bone-in-world (GlobalPose ``R_sim @ RBS.T``).

    Args:
        rot_sensor_world: (T, S, 3, 3) sensor orientations in world frame.
        rbs:              (S, 3, 3) bone-to-sensor mount.

    Returns:
        rot_bone_world: (T, S, 3, 3) bone orientations in world frame.
    """
    return rot_sensor_world @ rbs.unsqueeze(0).transpose(-1, -2)
