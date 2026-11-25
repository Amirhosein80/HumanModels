"""General-purpose signal filters for tensor sequences."""

from __future__ import annotations

import math

import torch

__all__ = ["savgol_filter"]

# Module-level cache for Savitzky-Golay coefficients.
# Keyed by (window_length, polyorder, deriv, delta, pos, dtype, device).
# The pseudoinverse computation is O(W^3) and identical across calls with the
# same parameters, so caching avoids redundant work on repeated filtering.
_COEFF_CACHE: dict[tuple, torch.Tensor] = {}


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

    Results are cached in :data:`_COEFF_CACHE` so repeated calls with the same
    parameters (e.g. filtering many channels with the same window) reuse the
    precomputed weights instead of recomputing the pseudoinverse.

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
    key = (window_length, polyorder, deriv, delta, pos, dtype, device)
    coeffs = _COEFF_CACHE.get(key)
    if coeffs is None:
        x = torch.arange(0, window_length, dtype=torch.float64) - pos  # (W,)
        A = torch.stack([x**j for j in range(polyorder + 1)], dim=1)  # (W, P+1)
        coeffs = torch.linalg.pinv(A)[deriv]  # (W,)
        coeffs = coeffs * (math.factorial(deriv) / delta**deriv)
        coeffs = coeffs.to(dtype=dtype, device=device)
        _COEFF_CACHE[key] = coeffs
    return coeffs


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
    ndim = x_m.ndim

    # Interior: centred convolution with nearest-pad (pos = half).
    # Vectorised over the window using advanced indexing instead of a Python loop.
    c_center = _savgol_coeffs(
        window_length, polyorder, deriv, delta, half, dtype, device
    )
    xp = torch.cat([x_m[[0] * half], x_m, x_m[[-1] * half]], dim=0)  # (T + W - 1, ...)
    idx = torch.arange(T, device=device).unsqueeze(1) + torch.arange(
        window_length, device=device
    ).unsqueeze(
        0
    )  # (T, W)
    gathered = xp[idx]  # (T, W, ...)
    c_view = c_center.view(window_length, *([1] * (ndim - 1)))  # (W, 1, ...)
    result = (gathered * c_view).sum(dim=1)  # (T, ...)

    # Left boundary: fit polynomial to x[0:W], evaluate at pos 0..half-1.
    # Stack all boundary coefficients and apply in one batched matmul.
    left_win = x_m[:window_length]  # (W, ...)
    left_win_flat = left_win.reshape(window_length, -1)  # (W, rest)
    left_coeffs = torch.stack(
        [
            _savgol_coeffs(window_length, polyorder, deriv, delta, i, dtype, device)
            for i in range(half)
        ]
    )  # (half, W)
    result[:half] = (left_coeffs @ left_win_flat).reshape(half, *x_m.shape[1:])

    # Right boundary: fit polynomial to x[T-W:T], evaluate at W-half..W-1.
    right_win = x_m[T - window_length :]  # (W, ...)
    right_win_flat = right_win.reshape(window_length, -1)  # (W, rest)
    right_coeffs = torch.stack(
        [
            _savgol_coeffs(
                window_length,
                polyorder,
                deriv,
                delta,
                window_length - half + i,
                dtype,
                device,
            )
            for i in range(half)
        ]
    )  # (half, W)
    result[T - half :] = (right_coeffs @ right_win_flat).reshape(half, *x_m.shape[1:])

    return result.movedim(0, axis)
