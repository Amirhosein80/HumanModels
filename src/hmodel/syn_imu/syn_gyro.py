"""Synthetic gyroscope synthesis via finite-difference angular velocity."""

from __future__ import annotations

import torch

__all__ = ["get_local_ang_vel_diff"]


def get_local_ang_vel_diff(
    rotation: torch.Tensor,
    fps: int,
) -> torch.Tensor:
    """Compute local angular velocity from a sequence of rotation matrices.

    Uses central finite differences in the interior and one-sided differences
    at the boundary frames.

    Args:
        rotation: (T, S, 3, 3) orientation matrices (world ← local) over time.
        fps:      Sampling rate in Hz.

    Returns:
        gyr: (T, S, 3) local-frame angular velocity in rad/s.
    """
    rotation_dot = torch.zeros_like(rotation)
    # Central difference for interior frames
    rotation_dot[1:-1] = rotation[2:] - rotation[:-2]
    # Forward difference at t=0, backward at t=T-1
    rotation_dot[0] = -3 * rotation[0] + 4 * rotation[1] - rotation[2]
    rotation_dot[-1] = 3 * rotation[-1] - 4 * rotation[-2] + rotation[-3]
    rotation_dot = rotation_dot * (fps / 2)

    # Skew-symmetric part of R^T dR/dt gives the angular velocity matrix
    ang_vel = rotation.transpose(-1, -2) @ rotation_dot
    ang_vel = (ang_vel - ang_vel.transpose(-1, -2)) / 2
    # Extract axial vector from skew-symmetric matrix
    return torch.stack(
        [ang_vel[..., 2, 1], ang_vel[..., 0, 2], ang_vel[..., 1, 0]], dim=-1
    )
