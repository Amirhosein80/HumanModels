"""Tests for hmodel.syn_imu — synthetic IMU synthesis from SMPL poses."""

from __future__ import annotations

import pytest
import torch

from hmodel.syn_imu.imu_calib import imu_calibration
from hmodel.syn_imu.syn_acc import get_local_acc_finite_diff, get_local_acc_sav_gol
from hmodel.syn_imu.syn_gyro import get_local_ang_vel_diff
from hmodel.syn_imu.syn_imu import SynIMU
from hmodel.syn_imu.syn_ori import integrate_ori_gyro

T = 60  # sequence length used across tests
S = 6  # number of sensors


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(params=["cpu"], ids=["cpu"])
def device(request):
    return torch.device(request.param)


@pytest.fixture(params=[torch.float32, torch.float64], ids=["f32", "f64"])
def dtype(request):
    return request.param


# ── get_local_acc_sav_gol ─────────────────────────────────────────────────────


class TestGetLocalAccSavGol:

    def test_output_shape(self, dtype, device):
        pos = torch.zeros(T, S, 3, dtype=dtype, device=device)
        rot = torch.eye(3, dtype=dtype, device=device).expand(T, S, 3, 3)
        gw = torch.tensor([0.0, -9.81, 0.0], dtype=dtype, device=device)
        out = get_local_acc_sav_gol(pos, rot, fps=60, gw=gw)
        assert out.shape == (T, S, 3)

    def test_dtype_propagation(self, dtype, device):
        pos = torch.zeros(T, S, 3, dtype=dtype, device=device)
        rot = torch.eye(3, dtype=dtype, device=device).expand(T, S, 3, 3)
        gw = torch.tensor([0.0, -9.81, 0.0], dtype=dtype, device=device)
        out = get_local_acc_sav_gol(pos, rot, fps=60, gw=gw)
        assert out.dtype == dtype

    def test_stationary_body_reads_gravity(self, dtype, device):
        """Stationary body (constant position, identity rotation) → acc ≈ -gw in local frame.

        For a stationary sensor: world acceleration a_world = 0.
        Local acc = R^T (a_world - gw) = -gw   (with identity R).
        """
        pos = torch.zeros(T, S, 3, dtype=dtype, device=device)
        rot = torch.eye(3, dtype=dtype, device=device).expand(T, S, 3, 3).clone()
        gw = torch.tensor([0.0, -9.81, 0.0], dtype=dtype, device=device)
        out = get_local_acc_sav_gol(pos, rot, fps=60, gw=gw)
        expected = -gw.expand(T, S, 3)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_constant_velocity_body(self, dtype, device):
        """Linearly moving body (zero second derivative) → acc ≈ -gw in local frame."""
        t = torch.linspace(0, 1, T, dtype=dtype, device=device)
        pos = t[:, None, None] * torch.ones(1, S, 3, dtype=dtype, device=device)
        rot = torch.eye(3, dtype=dtype, device=device).expand(T, S, 3, 3).clone()
        gw = torch.tensor([0.0, -9.81, 0.0], dtype=dtype, device=device)
        out = get_local_acc_sav_gol(pos, rot, fps=T - 1, gw=gw)
        # 2nd derivative of linear motion is zero → acc = -gw
        expected = -gw.expand(T, S, 3)
        # Allow looser tolerance because SG boundary effects are larger here
        torch.testing.assert_close(out[5:-5], expected[5:-5], atol=1e-3, rtol=1e-3)


# ── get_local_acc_finite_diff (paper FD replica) ──────────────────────────────


class TestGetLocalAccFiniteDiff:

    def test_output_shape(self, dtype, device):
        pos = torch.zeros(T, S, 3, dtype=dtype, device=device)
        rot = torch.eye(3, dtype=dtype, device=device).expand(T, S, 3, 3)
        gw = torch.tensor([0.0, -9.8, 0.0], dtype=dtype, device=device)
        out = get_local_acc_finite_diff(pos, rot, fps=60, gw=gw)
        assert out.shape == (T, S, 3)

    def test_stationary_body_reads_gravity(self, dtype, device):
        """Constant position → world acc 0 → local acc = -gw."""
        pos = torch.zeros(T, S, 3, dtype=dtype, device=device)
        rot = torch.eye(3, dtype=dtype, device=device).expand(T, S, 3, 3).clone()
        gw = torch.tensor([0.0, -9.8, 0.0], dtype=dtype, device=device)
        out = get_local_acc_finite_diff(pos, rot, fps=60, gw=gw)
        torch.testing.assert_close(out, -gw.expand(T, S, 3), atol=1e-4, rtol=1e-4)

    def test_short_sequence_raises(self, dtype, device):
        pos = torch.zeros(3, S, 3, dtype=dtype, device=device)
        rot = torch.eye(3, dtype=dtype, device=device).expand(3, S, 3, 3).clone()
        gw = torch.tensor([0.0, -9.8, 0.0], dtype=dtype, device=device)
        with pytest.raises(ValueError, match="T >= 4"):
            get_local_acc_finite_diff(pos, rot, fps=60, gw=gw)


# ── syn_ori (gyro integration) ────────────────────────────────────────────────


class TestSynOri:

    def test_integrate_static_gives_constant(self, dtype, device):
        """Zero gyro → integrated orientation stays at the initial frame."""
        torch.manual_seed(0)
        r0 = torch.randn(S, 3, 3, dtype=dtype, device=device)
        q, _ = torch.linalg.qr(r0)
        rot = q.unsqueeze(0).expand(T, S, 3, 3).clone()
        gyr = torch.zeros(T, S, 3, dtype=dtype, device=device)
        out = integrate_ori_gyro(rot, gyr, fps=60)
        torch.testing.assert_close(
            out, rot[0].unsqueeze(0).expand_as(out), atol=1e-5, rtol=1e-5
        )


# ── get_local_ang_vel_diff ────────────────────────────────────────────────────


class TestGetLocalAngVelDiff:

    def test_output_shape(self, dtype, device):
        rot = torch.eye(3, dtype=dtype, device=device).expand(T, S, 3, 3).clone()
        out = get_local_ang_vel_diff(rot, fps=60)
        assert out.shape == (T, S, 3)

    def test_dtype_propagation(self, dtype, device):
        rot = torch.eye(3, dtype=dtype, device=device).expand(T, S, 3, 3).clone()
        out = get_local_ang_vel_diff(rot, fps=60)
        assert out.dtype == dtype

    def test_constant_rotation_gives_zero(self, dtype, device):
        """Constant (all-identity) rotation sequence → angular velocity == 0."""
        rot = torch.eye(3, dtype=dtype, device=device).expand(T, S, 3, 3).clone()
        out = get_local_ang_vel_diff(rot, fps=60)
        torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-6, rtol=0.0)

    def test_rotation_about_y_gives_correct_omega(self, dtype, device):
        """Steady rotation about world Y at ω rad/s → gyr[:, :, 1] ≈ ω."""
        omega = 1.0  # rad/s
        fps = 100
        angles = torch.linspace(
            0.0, omega * (T - 1) / fps, T, dtype=dtype, device=device
        )
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        zero = torch.zeros(T, dtype=dtype, device=device)
        one = torch.ones(T, dtype=dtype, device=device)
        # Rotation matrix about Y axis: Ry(θ)
        Ry = torch.stack(
            [
                torch.stack([cos_a, zero, sin_a], dim=-1),
                torch.stack([zero, one, zero], dim=-1),
                torch.stack([-sin_a, zero, cos_a], dim=-1),
            ],
            dim=-2,
        )  # (T, 3, 3)
        rot = Ry[:, None].expand(T, S, 3, 3).clone()
        out = get_local_ang_vel_diff(rot, fps=fps)
        # Interior frames; boundary frames have larger FD error
        torch.testing.assert_close(
            out[5:-5, :, 1],
            torch.full((T - 10, S), omega, dtype=dtype, device=device),
            atol=1e-3,
            rtol=1e-3,
        )


# ── imu_calibration ───────────────────────────────────────────────────────────


class TestImuCalibration:

    def _identity_inputs(self, dtype, device):
        acc = torch.randn(T, S, 3, dtype=dtype, device=device)
        gyro = torch.randn(T, S, 3, dtype=dtype, device=device)
        rbw = torch.eye(3, dtype=dtype, device=device).expand(T, S, 3, 3).clone()
        rlw = torch.eye(3, dtype=dtype, device=device).expand(S, 3, 3).clone()
        rlc = torch.eye(3, dtype=dtype, device=device).expand(S, 3, 3).clone()
        return acc, gyro, rbw, rlw, rlc

    def test_output_keys(self, dtype, device):
        result = imu_calibration(*self._identity_inputs(dtype, device))
        assert set(result.keys()) == {"acc_calib", "gyro_calib", "rot_bone_calib"}

    def test_output_shapes(self, dtype, device):
        result = imu_calibration(*self._identity_inputs(dtype, device))
        assert result["acc_calib"].shape == (T, S, 3)
        assert result["gyro_calib"].shape == (T, S, 3)
        assert result["rot_bone_calib"].shape == (T, S, 3, 3)

    def test_identity_calibration_rot_unchanged(self, dtype, device):
        """Identity lab/calib rotations → rot_bone_calib == rot_bone_world."""
        acc, gyro, rbw, rlw, rlc = self._identity_inputs(dtype, device)
        result = imu_calibration(acc, gyro, rbw, rlw, rlc, remove_g=False)
        torch.testing.assert_close(result["rot_bone_calib"], rbw, atol=1e-6, rtol=0.0)

    def test_identity_calibration_gyro_unchanged(self, dtype, device):
        """Identity lab/calib rotations → gyro_calib == gyro."""
        acc, gyro, rbw, rlw, rlc = self._identity_inputs(dtype, device)
        result = imu_calibration(acc, gyro, rbw, rlw, rlc, remove_g=False)
        torch.testing.assert_close(result["gyro_calib"], gyro, atol=1e-6, rtol=0.0)

    def test_remove_g_adds_gravity_offset(self, dtype, device):
        """remove_g=True adds [0, -9.8, 0] to calibrated acc vs remove_g=False."""
        acc, gyro, rbw, rlw, rlc = self._identity_inputs(dtype, device)
        with_g = imu_calibration(acc, gyro, rbw, rlw, rlc, remove_g=True)["acc_calib"]
        without_g = imu_calibration(acc, gyro, rbw, rlw, rlc, remove_g=False)[
            "acc_calib"
        ]
        diff = with_g - without_g
        g_expected = torch.tensor([0.0, -9.8, 0.0], dtype=dtype, device=device)
        torch.testing.assert_close(
            diff, g_expected.expand_as(diff), atol=1e-5, rtol=0.0
        )


# ── SynIMU (requires SMPL; session-scoped fixture from conftest.py) ───────────


class TestSynIMU:

    @pytest.fixture
    def imu_gen(self, smpl_male):
        return SynIMU(smpl_male, fps=60)

    def _make_pose(self, T: int, dtype=torch.float64):
        return (
            torch.eye(3, dtype=dtype)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(T, 24, 3, 3)
            .clone()
        )

    def _make_trans(self, T: int, dtype=torch.float64):
        return torch.zeros(T, 3, dtype=dtype)

    def test_output_keys(self, imu_gen):
        pose = self._make_pose(T)
        trans = self._make_trans(T)
        data = imu_gen.syn_imu(pose, trans, add_noise=False)
        expected_keys = {
            "FPS",
            "RBW",
            "ACC",
            "GYR",
            "RLW",
            "RLC",
            "RSB",
            "Pose",
            "Trans",
            "Grot",
            "Joint",
        }
        assert set(data.keys()) == expected_keys

    def test_output_shapes(self, imu_gen):
        pose = self._make_pose(T)
        trans = self._make_trans(T)
        data = imu_gen.syn_imu(pose, trans, add_noise=False)
        assert data["RBW"].shape == (T, 6, 3, 3)
        assert data["ACC"].shape == (T, 6, 3)
        assert data["GYR"].shape == (T, 6, 3)
        assert data["RLW"].shape == (6, 3, 3)
        assert data["RLC"].shape == (6, 3, 3)
        assert data["RSB"].shape == (6, 3, 3)
        assert data["Pose"].shape == (T, 24, 3, 3)
        assert data["Trans"].shape == (T, 3)
        assert data["Grot"].shape == (T, 24, 3, 3)
        assert data["Joint"].shape == (T, 24, 3)

    def test_no_noise_deterministic(self, imu_gen):
        """add_noise=False must produce identical results on two calls."""
        pose = self._make_pose(T)
        trans = self._make_trans(T)
        d1 = imu_gen.syn_imu(pose, trans, add_noise=False)
        d2 = imu_gen.syn_imu(pose, trans, add_noise=False)
        torch.testing.assert_close(d1["ACC"], d2["ACC"], atol=0.0, rtol=0.0)
        torch.testing.assert_close(d1["GYR"], d2["GYR"], atol=0.0, rtol=0.0)

    def test_noise_differs_from_clean(self, imu_gen):
        """add_noise=True must produce output different from the clean version."""
        pose = self._make_pose(T)
        trans = self._make_trans(T)
        clean = imu_gen.syn_imu(pose, trans, add_noise=False)
        noisy = imu_gen.syn_imu(pose, trans, add_noise=True)
        assert not torch.allclose(clean["ACC"], noisy["ACC"])

    def test_dtype_propagation(self, imu_gen):
        """All tensor outputs must share the dtype of the input pose."""
        pose = self._make_pose(T, dtype=torch.float64)
        trans = self._make_trans(T, dtype=torch.float64)
        data = imu_gen.syn_imu(pose, trans, add_noise=False)
        for key in (
            "RBW",
            "ACC",
            "GYR",
            "RLW",
            "RLC",
            "RSB",
            "Pose",
            "Trans",
            "Grot",
            "Joint",
        ):
            assert data[key].dtype == torch.float64, f"{key} has wrong dtype"

    def test_fps_override(self, imu_gen):
        """fps kwarg must override default and be reflected in returned FPS."""
        pose = self._make_pose(T)
        trans = self._make_trans(T)
        data = imu_gen.syn_imu(pose, trans, add_noise=False, fps=120)
        assert data["FPS"] == 120

    def test_joint_centered_root_at_origin(self, imu_gen):
        """Joint positions relative to root: root joint (idx 0) must be zero."""
        pose = self._make_pose(T)
        trans = self._make_trans(T)
        data = imu_gen.syn_imu(pose, trans, add_noise=False)
        torch.testing.assert_close(
            data["Joint"][:, 0, :],
            torch.zeros(T, 3, dtype=torch.float64),
            atol=1e-6,
            rtol=0.0,
        )

    def test_rsb_eye_without_noise(self, imu_gen):
        """add_noise=False → mount RSB must be identity."""
        pose = self._make_pose(T)
        trans = self._make_trans(T)
        data = imu_gen.syn_imu(pose, trans, add_noise=False)
        eye = torch.eye(3, dtype=torch.float64).expand(6, 3, 3)
        torch.testing.assert_close(data["RSB"], eye, atol=1e-8, rtol=0.0)

    def test_rsb_valid_rotation_with_noise(self, imu_gen):
        """add_noise=True → RSB rows must be proper rotations (R^T R=I, det=+1)."""
        pose = self._make_pose(T)
        trans = self._make_trans(T)
        data = imu_gen.syn_imu(pose, trans, add_noise=True)
        rsb = data["RSB"]
        eye = torch.eye(3, dtype=torch.float64).expand(6, 3, 3)
        torch.testing.assert_close(
            rsb.transpose(-1, -2) @ rsb, eye, atol=1e-6, rtol=0.0
        )
        torch.testing.assert_close(
            torch.det(rsb),
            torch.ones(6, dtype=torch.float64),
            atol=1e-6,
            rtol=0.0,
        )

    def test_acc_mode_paper_shapes(self, imu_gen):
        """acc_mode='paper' must produce finite (T, 6, 3) readings."""
        pose = self._make_pose(T)
        trans = self._make_trans(T)
        data = imu_gen.syn_imu(pose, trans, add_noise=False, acc_mode="paper")
        assert data["ACC"].shape == (T, 6, 3)
        assert torch.isfinite(data["ACC"]).all()

    def test_acc_mode_invalid_raises(self, imu_gen):
        """Unknown acc_mode must raise ValueError."""
        pose = self._make_pose(T)
        trans = self._make_trans(T)
        with pytest.raises(ValueError, match="acc_mode"):
            imu_gen.syn_imu(pose, trans, add_noise=False, acc_mode="bogus")

    def test_ori_mode_invalid_raises(self, imu_gen):
        """Unknown ori_mode must raise ValueError."""
        pose = self._make_pose(T)
        trans = self._make_trans(T)
        with pytest.raises(ValueError, match="ori_mode"):
            imu_gen.syn_imu(pose, trans, add_noise=False, ori_mode="bogus")
