"""Synthetic accelerometer synthesis."""

from __future__ import annotations

import torch

from ..core.filters import savgol_filter  # noqa: F401

__all__ = ["get_local_acc_sav_gol", "get_local_acc_finite_diff"]


def get_local_acc_sav_gol(
    position: torch.Tensor,
    rotation: torch.Tensor,
    fps: int,
    gw: torch.Tensor,
) -> torch.Tensor:
    """Compute local accelerometer readings via Savitzky-Golay differentiation.

    Applies a 5-point cubic SG filter (mode='interp') to obtain the 2nd derivative
    of position (world-frame acceleration), subtracts gravity, then rotates into
    the sensor local frame.

    Args:
        position: (T, S, 3) sensor positions in world frame.
        rotation: (T, S, 3, 3) sensor orientation matrices (world <- local).
        fps:      Sampling rate in Hz.
        gw:       (3,) gravity vector in world frame (e.g. ``[0, -9.81, 0]``).

    Returns:
        acc: (T, S, 3) local-frame accelerometer readings.
    """
    acc = savgol_filter(
        position, window_length=5, polyorder=3, deriv=2, delta=1.0 / fps, axis=0
    )  # (T, S, 3) — world-frame acceleration
    acc = rotation.transpose(-1, -2) @ (acc - gw).unsqueeze(-1)
    return acc.squeeze(-1)


def get_local_acc_finite_diff(
    position: torch.Tensor,
    rotation: torch.Tensor,
    fps: int,
    gw: torch.Tensor,
) -> torch.Tensor:
    """Compute local accelerometer readings via 2nd-order finite differences.

    Faithful replica of GlobalPose ``IMUSimulator.get_acceleration``: central
    2nd difference in the interior, 4-point one-sided stencils at the
    boundaries, scaled by ``fps**2``, gravity subtracted, then rotated into
    the sensor local frame.

    Args:
        position: (T, S, 3) sensor positions in world frame (T >= 4).
        rotation: (T, S, 3, 3) sensor orientation matrices (world <- local).
        fps:      Sampling rate in Hz.
        gw:       (3,) gravity vector in world frame (e.g. ``[0, -9.8, 0]``).

    Returns:
        acc: (T, S, 3) local-frame accelerometer readings.
    """
    if position.shape[0] < 4:
        raise ValueError(
            f"finite-difference acceleration needs T >= 4 frames, got {position.shape[0]}"
        )
    a_mid = position[:-2] - 2 * position[1:-1] + position[2:]
    a0 = 2 * position[0] - 5 * position[1] + 4 * position[2] - position[3]
    a1 = 2 * position[-1] - 5 * position[-2] + 4 * position[-3] - position[-4]
    aW = torch.cat((a0.unsqueeze(0), a_mid, a1.unsqueeze(0)), dim=0)
    aW = aW * (fps * fps)  # (T, S, 3) — world-frame acceleration
    acc = rotation.transpose(-1, -2) @ (aW - gw).unsqueeze(-1)
    return acc.squeeze(-1)
