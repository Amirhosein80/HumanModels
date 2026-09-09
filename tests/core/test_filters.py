"""Tests for hmodel.core.filters — savgol_filter."""

from __future__ import annotations

import pytest
import torch
from scipy.signal import savgol_filter as scipy_savgol

from hmodel.core.filters import savgol_filter

T = 60
S = 6


@pytest.fixture(
    params=[torch.float32, torch.float64, torch.float16, torch.bfloat16],
    ids=["f32", "f64", "f16", "bf16"],
)
def dtype(request):
    return request.param


class TestSavgolFilter:
    """Validate the from-scratch savgol_filter against scipy's implementation."""

    @pytest.mark.parametrize(
        "deriv,delta",
        [
            (0, 1.0),
            (1, 1 / 60),
            (2, 1 / 60),
        ],
    )
    def test_matches_scipy_interp_axis0(self, deriv, delta):
        """Must match scipy's default mode='interp' (polynomial boundary fit)."""
        torch.manual_seed(0)
        x = torch.randn(T, S, 3, dtype=torch.float64)
        got = savgol_filter(
            x, window_length=5, polyorder=3, deriv=deriv, delta=delta, axis=0
        )
        ref = torch.from_numpy(
            scipy_savgol(
                x.numpy(),
                window_length=5,
                polyorder=3,
                deriv=deriv,
                delta=delta,
                axis=0,
            )
        )
        torch.testing.assert_close(got, ref, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_matches_scipy_interp_various_axes(self, axis):
        """axis parameter must redirect the filter correctly."""
        torch.manual_seed(1)
        x = torch.randn(T, S, 3, dtype=torch.float64)
        got = savgol_filter(x, window_length=5, polyorder=3, deriv=0, axis=axis)
        ref = torch.from_numpy(
            scipy_savgol(x.numpy(), window_length=5, polyorder=3, deriv=0, axis=axis)
        )
        torch.testing.assert_close(got, ref, atol=1e-10, rtol=1e-10)

    def test_dtype_propagation(self, dtype):
        x = torch.randn(T, S, 3, dtype=dtype)
        out = savgol_filter(x, window_length=5, polyorder=3, deriv=2, delta=1 / 60)
        assert out.dtype == dtype

    def test_output_shape(self):
        x = torch.randn(T, S, 3, dtype=torch.float64)
        out = savgol_filter(x, window_length=5, polyorder=3, deriv=2, delta=1 / 60)
        assert out.shape == x.shape
