"""General-purpose signal filters for tensor sequences."""

from __future__ import annotations

import math

import torch

__all__ = ["savgol_filter"]


def _savgol_coeffs(
    window_length: int,
    polyorder: int,
    deriv: int,
    delta: float,
    pos: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Savitzky-Golay convolution weights via Vandermonde pseudoinverse.

    Matches ``scipy.signal.savgol_coeffs(window_length, polyorder, deriv, delta, pos)``.

    Args:
        window_length: Number of samples in the filter window (must be odd).
        polyorder:     Polynomial order to fit.
        deriv:         Order of the derivative to compute (0 = smoothing).
        delta:         Sample spacing (e.g. ``1/fps`` for time-series data).
        pos:           Position within the window [0, window_length-1] at which the
                       polynomial is evaluated. Use ``window_length // 2`` for the
                       centred (interior) case.
        dtype:         Output dtype.
        device:        Output device.

    Returns:
        coeffs: (window_length,) convolution weights.
    """
    x = torch.arange(0, window_length, dtype=torch.float64) - pos  # (W,)
    A = torch.stack([x**j for j in range(polyorder + 1)], dim=1)  # (W, P+1)
    coeffs = torch.linalg.pinv(A)[deriv]  # (W,)
    coeffs = coeffs * (math.factorial(deriv) / delta**deriv)
    return coeffs.to(dtype=dtype, device=device)


def savgol_filter(
    x: torch.Tensor,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    axis: int = 0,
) -> torch.Tensor:
    """Savitzky-Golay filter — PyTorch implementation of ``scipy.signal.savgol_filter``.

    Matches scipy's default ``mode='interp'``:
    - Interior points use the standard centred SG convolution.
    - The first and last ``window_length // 2`` points are computed by fitting a
      degree-``polyorder`` polynomial to the nearest ``window_length`` edge samples
      and evaluating the ``deriv``-th derivative at the boundary position.

    Args:
        x:             Input tensor, arbitrary shape.
        window_length: Filter window size (must be odd and > polyorder).
        polyorder:     Polynomial order.
        deriv:         Derivative order (0 = smooth, 1 = velocity, 2 = acceleration).
        delta:         Sample spacing used when computing derivatives.
        axis:          Axis along which to apply the filter.

    Returns:
        Filtered tensor, same shape and dtype as ``x``.
    """
    half = window_length // 2
    dtype, device = x.dtype, x.device

    x_m = x.movedim(axis, 0)  # (T, ...)
    T = x_m.shape[0]

    # Interior: centred convolution with nearest-pad (pos = half)
    c_center = _savgol_coeffs(
        window_length, polyorder, deriv, delta, half, dtype, device
    )
    xp = torch.cat([x_m[[0] * half], x_m, x_m[[-1] * half]], dim=0)
    result = sum(c_center[i] * xp[i : i + T] for i in range(window_length))

    # Left boundary: fit polynomial to x[0:W], evaluate at pos 0..half-1
    left_win = x_m[:window_length]
    for i in range(half):
        c = _savgol_coeffs(window_length, polyorder, deriv, delta, i, dtype, device)
        result[i] = torch.tensordot(c, left_win, dims=([0], [0]))

    # Right boundary: fit polynomial to x[T-W:T], evaluate at W-half..W-1
    right_win = x_m[T - window_length :]
    for i in range(half):
        pos = window_length - half + i
        c = _savgol_coeffs(window_length, polyorder, deriv, delta, pos, dtype, device)
        result[T - half + i] = torch.tensordot(c, right_win, dims=([0], [0]))

    return result.movedim(0, axis)
