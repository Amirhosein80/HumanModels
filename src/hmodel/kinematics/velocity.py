"""Velocity estimation from position sequences."""

from __future__ import annotations

import torch

from ..core.filters import savgol_filter

__all__ = ["get_velocity", "get_realtime_velocity"]


def get_velocity(
    position: torch.Tensor,
    fps: int | float,
    axis: int = 0,
    window_length: int = 5,
    polyorder: int = 3,
) -> torch.Tensor:
    """Estimate velocity via Savitzky-Golay 1st derivative (mode='interp').

    Uses a Savitzky-Golay polynomial fit to compute the 1st derivative of
    ``position`` along ``axis``, matching ``scipy.signal.savgol_filter`` with
    the default ``mode='interp'`` boundary handling.

    Args:
        position:      Tensor of positions, arbitrary shape.
        fps:           Sampling rate in Hz (used as ``1/delta``).
        axis:          Time axis along which to differentiate.
        window_length: SG window size (must be odd and > polyorder).
        polyorder:     Polynomial order for the local fit.

    Returns:
        velocity: Same shape and dtype as ``position``.
    """
    return savgol_filter(
        position,
        window_length=window_length,
        polyorder=polyorder,
        deriv=1,
        delta=1.0 / fps,
        axis=axis,
    )


def get_realtime_velocity(
    position0: torch.Tensor,
    position_1: torch.Tensor,
    position_2: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Real-time (causal) velocity via 2nd-order backward finite difference.

    Computes velocity at the current time step using only the current and two
    preceding position samples — no future samples required.

    Formula:  v(t) = [3·x(t) - 4·x(t-1) + x(t-2)] / (2·dt)

    Args:
        position0:  x(t)   — current position, arbitrary shape (...).
        position_1: x(t-1) — position one step back, same shape as position0.
        position_2: x(t-2) — position two steps back, same shape as position0.
        dt:         Sample spacing in seconds (e.g. ``1/fps``).

    Returns:
        velocity: Same shape and dtype as ``position0``.
    """
    return (3.0 * position0 - 4.0 * position_1 + position_2) / (2.0 * dt)
