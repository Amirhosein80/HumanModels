"""Tests for the quarantined STGIPoser TotalCapture port (PIP-stencil inside)."""

from __future__ import annotations

import pytest
import torch

from external.stgiposer_totalcapture import (
    prepare_motion,
    step_for_framerate,
    syn_acc,
    syn_uwb,
    syn_vel,
    synthesize_tc_sequence,
)
from external.stgiposer_totalcapture.synth_ops import TC_IMU_MASK

T = 24  # frames (>= 9 for the wide stencil)
S = 6


def _syn_acc_literal(trans: torch.Tensor, fps: float = 60.0) -> torch.Tensor:
    """Loop transcription of the upstream wide-stencil formula."""
    scale = fps * fps
    acc = torch.zeros_like(trans)
    for t in range(1, trans.shape[0] - 1):
        acc[t] = (trans[t - 1] + trans[t + 1] - 2 * trans[t]) * scale
    for t in range(4, trans.shape[0] - 4):
        acc[t] = (trans[t - 4] + trans[t + 4] - 2 * trans[t]) * scale / 16
    return acc


class TestSynAcc:
    def test_matches_literal_stencil(self):
        torch.manual_seed(0)
        pos = torch.randn(T, S, 3, dtype=torch.float64) * 2
        torch.testing.assert_close(
            syn_acc(pos), _syn_acc_literal(pos), atol=1e-9, rtol=1e-9
        )

    def test_borders_stay_zero(self):
        torch.manual_seed(1)
        pos = torch.randn(T, S, 3, dtype=torch.float64)
        out = syn_acc(pos)
        torch.testing.assert_close(out[0], torch.zeros_like(out[0]))
        torch.testing.assert_close(out[-1], torch.zeros_like(out[-1]))

    def test_fps_scaling(self):
        torch.manual_seed(2)
        pos = torch.randn(T, S, 3, dtype=torch.float64)
        out60 = syn_acc(pos, fps=60.0)
        out120 = syn_acc(pos, fps=120.0)
        torch.testing.assert_close(out120[4:-4], 4 * out60[4:-4], atol=1e-8, rtol=1e-8)

    def test_short_sequence_raises(self):
        with pytest.raises(ValueError, match="T >= 9"):
            syn_acc(torch.zeros(8, S, 3))


class TestSynVel:
    def test_constant_velocity_eye_root(self):
        d = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
        trans = torch.stack([i * d for i in range(T)]).unsqueeze(1)  # (T, 1, 3)
        root = torch.eye(3, dtype=torch.float64).expand(T, 1, 3, 3)
        out = syn_vel(trans, root)
        torch.testing.assert_close(out[0], torch.zeros(1, 3, dtype=torch.float64))
        torch.testing.assert_close(
            out[1:], (d * 60.0).expand(T - 1, 1, 3), atol=1e-9, rtol=1e-9
        )


class TestSynUwb:
    def test_shape_symmetry_zero_diag(self):
        torch.manual_seed(3)
        p = torch.randn(T, S, 3, dtype=torch.float64)
        out = syn_uwb(p)
        assert out.shape == (T, S, S)
        torch.testing.assert_close(out, out.transpose(-1, -2))
        torch.testing.assert_close(
            out.diagonal(dim1=-2, dim2=-1), torch.zeros(T, S, dtype=torch.float64)
        )


class TestMasksAndSteps:
    def test_tc_imu_mask_is_permutation(self):
        assert sorted(TC_IMU_MASK) == [0, 1, 2, 3, 4, 5]

    def test_step_map(self):
        assert step_for_framerate(60) == 1
        assert step_for_framerate(120) == 2
        assert step_for_framerate(250) == 5
        assert step_for_framerate(150) == 3

    def test_step_unsupported_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            step_for_framerate(30)


class TestPrepareMotion:
    def test_root_relative_trans_and_shapes(self):
        torch.manual_seed(4)
        poses_52 = torch.randn(T, 52, 3, dtype=torch.float64) * 0.1
        trans = torch.randn(T, 3, dtype=torch.float64) * 5 + 10
        pose_24, tout, step = prepare_motion(poses_52, trans, framerate=60)
        assert step == 1
        assert pose_24.shape == (T, 24, 3)
        assert tout.shape == (T, 3)
        torch.testing.assert_close(
            tout[0], torch.zeros(3, dtype=torch.float64), atol=1e-9, rtol=0.0
        )


class TestSynthesizeSmoke:
    def test_keys_shapes_bias_match(self, smpl_male):
        torch.manual_seed(5)
        n = T
        acc = torch.randn(n, S, 3, dtype=torch.float64)
        ori = torch.eye(3, dtype=torch.float64).expand(n, S, 3, 3).clone()
        poses_52 = torch.randn(n, 52, 3, dtype=torch.float64) * 0.2
        trans = torch.randn(n, 3, dtype=torch.float64)
        out = synthesize_tc_sequence(acc, ori, poses_52, trans, 60.0, smpl_male)
        assert out["imu_acc"].shape == (n, S, 3)
        assert out["imu_ori"].shape == (n, S, 3, 3)
        assert out["pose"].shape == (n, 24, 3, 3)
        assert out["grot"].shape == (n, 24, 3, 3)
        assert out["jvel"].shape == (n, 24, 3)
        assert out["uwb"].shape == (n, S, S)
        assert out["vacc"].shape == (n, S, 3)
        for v in out.values():
            assert torch.isfinite(v).all()
        # Bias-match shift is constant over time and reported exactly.
        shift = out["imu_acc"] - out["imu_acc_raw"]
        torch.testing.assert_close(
            shift, out["acc_bias_shift"].unsqueeze(0).expand_as(shift)
        )
