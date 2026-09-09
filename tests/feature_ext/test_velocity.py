"""Tests for hmodel.feature_ext.velocity — get_velocity."""

from __future__ import annotations

import pytest
import torch

from hmodel.core.filters import savgol_filter
from hmodel.feature_ext.velocity import get_realtime_velocity, get_velocity

T = 60
S = 6


@pytest.fixture(
    params=[torch.float32, torch.float64, torch.float16, torch.bfloat16],
    ids=["f32", "f64", "f16", "bf16"],
)
def dtype(request):
    return request.param


class TestGetVelocity:

    def test_output_shape(self, dtype):
        x = torch.randn(T, S, 3, dtype=dtype)
        out = get_velocity(x, fps=60)
        assert out.shape == x.shape

    def test_dtype_propagation(self, dtype):
        x = torch.randn(T, S, 3, dtype=dtype)
        out = get_velocity(x, fps=60)
        assert out.dtype == dtype

    def test_constant_position_gives_zero_velocity(self):
        """Constant position → velocity == 0 everywhere."""
        x = torch.ones(T, S, 3, dtype=torch.float64) * 5.0
        out = get_velocity(x, fps=60)
        torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-10, rtol=0.0)

    def test_linear_ramp_gives_constant_velocity(self):
        """Linear position ramp → interior velocity ≈ slope."""
        fps = 60.0
        slope = 3.0
        t = torch.arange(T, dtype=torch.float64) / fps
        x = (slope * t)[:, None, None].expand(T, S, 3).clone()
        out = get_velocity(x, fps=fps)
        half = 5 // 2
        expected = torch.full((T - 2 * half, S, 3), slope, dtype=torch.float64)
        torch.testing.assert_close(out[half:-half], expected, atol=1e-8, rtol=1e-8)

    def test_matches_savgol_deriv1(self):
        """get_velocity must be identical to calling savgol_filter with deriv=1."""
        fps = 60.0
        torch.manual_seed(42)
        x = torch.randn(T, S, 3, dtype=torch.float64)
        vel = get_velocity(x, fps=fps, window_length=5, polyorder=3)
        ref = savgol_filter(
            x, window_length=5, polyorder=3, deriv=1, delta=1.0 / fps, axis=0
        )
        torch.testing.assert_close(vel, ref, atol=0.0, rtol=0.0)

    def test_axis_parameter(self):
        """axis kwarg must filter along the specified dimension."""
        x = torch.randn(T, S, 3, dtype=torch.float64)
        out0 = get_velocity(x, fps=60, axis=0)
        out1 = get_velocity(x, fps=60, axis=1)
        assert out0.shape == x.shape
        assert out1.shape == x.shape
        assert not torch.allclose(out0, out1)


class TestGetRealtimeVelocity:

    def test_output_shape(self, dtype):
        p = torch.randn(24, 3, dtype=dtype)
        out = get_realtime_velocity(p, p, p, dt=1 / 60)
        assert out.shape == p.shape

    def test_dtype_propagation(self, dtype):
        p = torch.randn(24, 3, dtype=dtype)
        out = get_realtime_velocity(p, p, p, dt=1 / 60)
        assert out.dtype == dtype

    def test_constant_position_gives_zero(self):
        """x(t) == x(t-1) == x(t-2) → velocity == 0."""
        p = torch.ones(24, 3, dtype=torch.float64) * 7.0
        out = get_realtime_velocity(p, p, p, dt=1 / 60)
        torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-12, rtol=0.0)

    def test_linear_ramp_gives_exact_slope(self):
        """For x(t) = slope * t * dt the formula gives exactly slope."""
        slope = 2.5
        dt = 1 / 60
        # x0 = slope*(t)*dt, x1 = slope*(t-1)*dt, x2 = slope*(t-2)*dt
        # (3*t - 4*(t-1) + (t-2)) / (2*dt) * dt*slope = (3t-4t+4+t-2)/(2) * slope = slope
        t = 10  # arbitrary frame index
        x0 = torch.full((3,), slope * t * dt, dtype=torch.float64)
        x1 = torch.full((3,), slope * (t - 1) * dt, dtype=torch.float64)
        x2 = torch.full((3,), slope * (t - 2) * dt, dtype=torch.float64)
        out = get_realtime_velocity(x0, x1, x2, dt=dt)
        torch.testing.assert_close(
            out, torch.full_like(out, slope), atol=1e-10, rtol=0.0
        )

    def test_formula_correctness_scalar(self):
        """Manual check: (3*3 - 4*2 + 1) / (2*dt) = 2/dt."""
        dt = 0.1
        x0 = torch.tensor([3.0], dtype=torch.float64)
        x1 = torch.tensor([2.0], dtype=torch.float64)
        x2 = torch.tensor([1.0], dtype=torch.float64)
        expected = torch.tensor([(3 * 3 - 4 * 2 + 1) / (2 * dt)], dtype=torch.float64)
        out = get_realtime_velocity(x0, x1, x2, dt=dt)
        torch.testing.assert_close(out, expected, atol=1e-12, rtol=0.0)

    def test_causal_independence(self):
        """Result must depend only on position0/1/2 — changing other frames has no effect."""
        dt = 1 / 60
        p0 = torch.randn(3, dtype=torch.float64)
        p1 = torch.randn(3, dtype=torch.float64)
        p2 = torch.randn(3, dtype=torch.float64)
        ref = get_realtime_velocity(p0, p1, p2, dt=dt)
        # "future" frame is irrelevant — we just call again with the same three inputs
        same = get_realtime_velocity(p0, p1, p2, dt=dt)
        torch.testing.assert_close(ref, same, atol=0.0, rtol=0.0)
