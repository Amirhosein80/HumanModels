"""Tests for hmodel.datasets.totalcapture (self-contained, no data files)."""

from __future__ import annotations

import pytest
import torch

from hmodel.datasets.totalcapture import (
    FILE_IDX,
    REPO_JOINT_MASK,
    SENSOR_NAMES,
    calibrate,
    load_amass_npz,
    load_calib,
    orientation_error_deg,
    parse_sensors,
    prepare_motion,
    select_sensors,
    stack_calib,
    step_for_framerate,
)

T = 16
S = 3


def _write_sensors_file(path, n_frames=4, n_sensors=2):
    lines = [f"{n_sensors} {n_frames}"]
    for f in range(n_frames):
        lines.append(str(f + 1))
        for s in range(n_sensors):
            lines.append(f"sens{s} 1 0 0 0 0 0 9.8 0 0 0 1 0 0")
    path.write_text("\n".join(lines) + "\n")


class TestParseSensors:
    def test_shapes_names_orthonormal(self, tmp_path):
        p = tmp_path / "x.sensors"
        _write_sensors_file(p)
        out = parse_sensors(str(p))
        assert out["names"] == ["sens0", "sens1"]
        assert out["RIS"].shape == (4, 2, 3, 3)
        assert out["aS"].shape == (4, 2, 3)
        eye = torch.eye(3, dtype=torch.float64).expand(4, 2, 3, 3)
        torch.testing.assert_close(out["RIS"].transpose(-1, -2) @ out["RIS"], eye)
        # identity quat → identity rotation; acc line was (0, 0, 9.8)
        torch.testing.assert_close(
            out["aS"],
            torch.tensor([0.0, 0.0, 9.8], dtype=torch.float64).expand(4, 2, 3),
        )

    def test_bad_frame_index_raises(self, tmp_path):
        p = tmp_path / "bad.sensors"
        p.write_text(
            "1 2\n1\ns0 1 0 0 0 0 0 0 0 0 0 0 0 0\n3\ns0 1 0 0 0 0 0 0 0 0 0 0 0 0\n"
        )
        with pytest.raises(ValueError, match="frame index"):
            parse_sensors(str(p))

    def test_quat_wxyz_to_rotmat_order(self, tmp_path):
        """File wxyz 90°-about-Z must decode to Rz(90°), pinning roma's xyzw order."""
        c, s = 0.7071067811865476, 0.7071067811865475
        p = tmp_path / "rot.sensors"
        p.write_text(f"1 1\n1\ns0 {c} 0 0 {s} 0 0 0 0 0 0 0 0 0\n")
        out = parse_sensors(str(p))
        expected = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(out["RIS"][0, 0], expected, atol=1e-6, rtol=0.0)

    def test_repo_order_constants(self):
        assert len(SENSOR_NAMES) == 6
        assert sorted(FILE_IDX) == [0, 2, 5, 6, 9, 10]
        assert REPO_JOINT_MASK == [0, 4, 5, 15, 18, 19]


class TestLoadCalib:
    def test_identity_quat_gives_fix(self, tmp_path):
        p = tmp_path / "c.txt"
        p.write_text("2\nPelvis 0 0 0 1\nHead 0 0 0 1\n")
        fix = torch.eye(3, dtype=torch.float64)
        out = load_calib(str(p), fix)
        torch.testing.assert_close(out["Pelvis"], fix)
        stacked = stack_calib(out, ["Pelvis", "Head"])
        assert stacked.shape == (2, 3, 3)

    def test_missing_sensor_raises(self):
        with pytest.raises(ValueError, match="missing sensor"):
            stack_calib({}, ["Pelvis"])

    def test_quat_xyzw_order_with_transpose(self, tmp_path):
        """File xyzw 90°-about-Z with eye fix must give Rz(90°)^T (upstream .t())."""
        c, s = 0.7071067811865476, 0.7071067811865475
        p = tmp_path / "c.txt"
        p.write_text(f"1\nPelvis 0 0 {s} {c}\n")
        out = load_calib(str(p), torch.eye(3, dtype=torch.float64))
        expected = torch.tensor(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(out["Pelvis"], expected, atol=1e-6, rtol=0.0)


class TestSelectSensors:
    def test_reorders_vectors_and_rotmats(self):
        n_file = 13  # full TotalCapture file sensor count
        v = torch.arange(T * n_file * 3, dtype=torch.float64).reshape(T, n_file, 3)
        m = torch.arange(T * n_file * 9, dtype=torch.float64).reshape(T, n_file, 3, 3)
        out_v = select_sensors(v, FILE_IDX)
        out_m = select_sensors(m, FILE_IDX)
        assert out_v.shape == (T, 6, 3)
        assert out_m.shape == (T, 6, 3, 3)
        torch.testing.assert_close(out_v[:, 0], v[:, 2])
        torch.testing.assert_close(out_m[:, 0], m[:, 2])


class TestStepAndPrepare:
    def test_step_map(self):
        assert step_for_framerate(60) == 1
        assert step_for_framerate(120) == 2
        assert step_for_framerate(250) == 5
        assert step_for_framerate(150) == 3
        with pytest.raises(ValueError, match="not supported"):
            step_for_framerate(30)

    def test_prepare_shapes_root_relative(self):
        torch.manual_seed(0)
        poses_52 = torch.randn(T, 52, 3, dtype=torch.float64) * 0.1
        trans = torch.randn(T, 3, dtype=torch.float64) * 5 + 10
        pose_24, tout = prepare_motion(poses_52, trans, framerate=60)
        assert pose_24.shape == (T, 24, 3)
        assert tout.shape == (T, 3)
        torch.testing.assert_close(
            tout[0], torch.zeros(3, dtype=torch.float64), atol=1e-9, rtol=0.0
        )

    def test_prepare_steps_120hz(self):
        poses_52 = torch.zeros(20, 52, 3, dtype=torch.float64)
        trans = torch.zeros(20, 3, dtype=torch.float64)
        pose_24, tout = prepare_motion(poses_52, trans, framerate=120)
        assert pose_24.shape == (10, 24, 3)

    def test_prepare_wrong_joints_raises(self):
        with pytest.raises(ValueError, match="52"):
            prepare_motion(torch.zeros(T, 24, 3), torch.zeros(T, 3), 60.0)

    def test_load_amass_npz_roundtrip(self, tmp_path):
        import numpy as np

        p = tmp_path / "a_poses.npz"
        np.savez(
            p,
            poses=np.zeros((8, 156)),
            trans=np.zeros((8, 3)),
            mocap_framerate=60.0,
            gender="male",
        )
        out = load_amass_npz(str(p))
        assert out["poses_52"].shape == (8, 52, 3)
        assert out["gender"] == "male"


class TestCalibrate:
    def test_identity_maps_to_testpy_convention(self):
        torch.manual_seed(1)
        acc = torch.randn(T, S, 3, dtype=torch.float64)
        gyr = torch.randn(T, S, 3, dtype=torch.float64)
        ris = torch.eye(3, dtype=torch.float64).expand(T, S, 3, 3).clone()
        rim = torch.eye(3, dtype=torch.float64).expand(S, 3, 3).clone()
        rsb = torch.eye(3, dtype=torch.float64).expand(S, 3, 3).clone()
        out = calibrate(acc, gyr, ris, rim, rsb)
        torch.testing.assert_close(out["rot_bone"], ris)
        torch.testing.assert_close(out["gyro"], gyr)
        g = torch.tensor([0.0, -9.8, 0.0], dtype=torch.float64)
        torch.testing.assert_close(out["acc"], acc + g.expand_as(acc))


class TestOrientationErrorDeg:
    def test_identical_is_zero(self):
        r = torch.eye(3, dtype=torch.float64).expand(T, 3, 3)
        torch.testing.assert_close(
            orientation_error_deg(r, r), torch.zeros(T, dtype=torch.float64)
        )

    def test_quarter_turn(self):
        a = torch.eye(3, dtype=torch.float64)
        b = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        )
        torch.testing.assert_close(
            orientation_error_deg(a, b), torch.tensor(90.0, dtype=torch.float64)
        )
