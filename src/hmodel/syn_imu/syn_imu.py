"""Synthetic IMU data generation from SMPL body model poses."""

from __future__ import annotations

import math

import roma
import torch

from ..bmodels.smpl import SMPL
from .syn_acc import get_local_acc_finite_diff, get_local_acc_sav_gol
from .syn_gyro import get_local_ang_vel_diff
from .syn_ori import add_ori_noise, integrate_ori_gyro, unmount_to_bone

__all__ = ["SynIMU"]

# Sensor placement: 6 IMU sites (pelvis, L/R knee, head, L/R wrist)
# NOTE: pelvis-first order is this repo's canonical convention (kept
# deliberately, differs from GlobalPose's pelvis-last order).
_VERTEX_MASK = [3021, 1176, 4662, 411, 1961, 5424]
_JOINT_MASK = [0, 4, 5, 15, 18, 19]
N_SENSORS = 6

# Gravity in SMPL coords (Y-up): matches GlobalPose gW = (0, -9.8, 0).
_GRAVITY = (0.0, -9.8, 0.0)

# Post-simulation measurement noise (GlobalPose _syn_imu defaults).
_ACC_WHITE = 5e-2
_ACC_WALK = 1e-4
_GYR_WHITE = 5e-3
_GYR_WALK = 1e-5


class SynIMU:
    """Synthesise virtual IMU sensor readings from SMPL kinematic sequences.

    Sensor positions are taken from vertices on the body surface; orientations
    follow the corresponding joint rotation.  Optional noise models realistic
    IMU drift (random-walk on position/orientation perturbations).

    Args:
        smpl: Loaded :class:`~hmodel.bmodels.smpl.SMPL` instance.
        fps:  Default sampling rate in Hz (can be overridden per call).
    """

    def __init__(self, smpl: SMPL, fps: int = 60) -> None:
        self.smpl = smpl
        self.fps = fps

        k = math.sqrt(math.pi / 8)
        self.k = k
        # Trajectory perturbation scales (GlobalPose _syn_imu):
        # position walk 1e-3*k, orientation walk 1e-2*k (10x larger).
        self.std_dp = 1e-3 * k * math.sqrt(1.0 / 60)
        self.std_dw = 1e-2 * k * math.sqrt(1.0 / 60)
        self.std = self.std_dp  # back-compat alias

        # Gravity in SMPL coords: Y-up → g points in -Y
        self.gw = torch.tensor(_GRAVITY, dtype=torch.float64)

    def walking_noise(
        self,
        shape: tuple[int, ...],
        std: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Random-walk noise: cumulative sum of Gaussian increments.

        Args:
            shape:  Output shape.
            std:    Standard deviation of each increment.
            dtype:  Output dtype.
            device: Output device.

        Returns:
            Noise tensor of the given shape.
        """
        return torch.cumsum(torch.randn(shape, dtype=dtype, device=device) * std, dim=0)

    @staticmethod
    def _random_rotation_matrix(
        n: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Uniform random rotation matrices via QR (replica of articulate's
        ``generate_random_rotation_matrix``).

        Args:
            n:      Number of matrices.
            dtype:  Output dtype (computation upcasts half precision).
            device: Output device.

        Returns:
            R: (n, 3, 3) rotation matrices with det == +1.
        """
        work_dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float32
        a = torch.randn(n, 3, 3, dtype=work_dtype, device=device)
        q, r = torch.linalg.qr(a)
        d = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
        d[d == 0] = 1.0
        q = q * d.unsqueeze(-2)
        neg = torch.det(q) < 0
        if bool(neg.any()):
            q[neg, :, 2] = -q[neg, :, 2]
        return q.to(dtype=dtype)

    def syn_imu(
        self,
        pose: torch.Tensor,
        translation: torch.Tensor,
        add_noise: bool = True,
        fps: int | None = None,
        acc_mode: str = "savgol",
        ori_mode: str = "integrated",
    ) -> dict:
        """Synthesise IMU readings for a kinematic sequence.

        Replicates GlobalPose ``_syn_imu`` with ``skip_ESKF=True`` (gyro
        integration): random-walk trajectory perturbations, bone-to-sensor
        mount offset (``RBS``), sensor-local acc/gyro synthesis, white +
        random-walk measurement noise, gyro integration with post-hoc
        orientation noise, and unmounting to bone-in-world ``RBW``.

        Args:
            pose:        (T, 24, 3, 3) local joint rotation matrices.
            translation: (T, 3) root position in metres.
            add_noise:   If True, add random-walk perturbations to position and
                         orientation before computing acc/gyro, plus
                         measurement/orientation noise after.
            fps:         Sampling rate override; defaults to ``self.fps``.
            acc_mode:    ``"savgol"`` (5-point cubic SG 2nd derivative,
                         this repo's default) or ``"paper"`` (2nd-order
                         finite differences, GlobalPose replica).
            ori_mode:    ``"integrated"`` (gyro integration). Kept for
                         forward-compatibility; only this mode is supported.

        Returns:
            dict with keys:
                ``FPS``   int
                ``RBW``   (T, 6, 3, 3) bone rotation in world frame
                ``ACC``   (T, 6, 3) sensor-local accelerometer (m/s²)
                ``GYR``   (T, 6, 3) sensor-local gyroscope (rad/s)
                ``RLW``   (6, 3, 3) lab-to-world rotation (eye: lab == world)
                ``RLC``   (6, 3, 3) lab-to-calib rotation (eye)
                ``RSB``   (6, 3, 3) bone-to-sensor mount (eye if no noise)
                ``Pose``  (T, 24, 3, 3)
                ``Trans`` (T, 3)
                ``Grot``  (T, 24, 3, 3) global joint rotations
                ``Joint`` (T, 24, 3) joint positions relative to root
        """
        if acc_mode not in ("savgol", "paper"):
            raise ValueError(f"acc_mode must be 'savgol' or 'paper', got {acc_mode!r}")
        if ori_mode != "integrated":
            raise ValueError(f"ori_mode must be 'integrated', got {ori_mode!r}")
        fps = fps if fps is not None else self.fps
        T = pose.shape[0]
        dtype, device = pose.dtype, pose.device

        all_grot, all_joint, all_vertex = self.smpl.forward_kinematics(
            pose=pose, root_position=translation, return_mesh=True
        )

        grot = all_grot[:, _JOINT_MASK].clone()  # (T, 6, 3, 3)
        vertex = all_vertex[:, _VERTEX_MASK].clone()  # (T, 6, 3)
        all_grot_out = all_grot  # keep ref before del
        del all_vertex  # free (T, V, 3) — largest allocation

        if add_noise:
            import math as _math

            dt_scale = _math.sqrt(1.0 / fps)
            std_dp = 1e-3 * self.k * dt_scale
            std_dw = 1e-2 * self.k * dt_scale
            dp = (
                self.walking_noise((T, N_SENSORS, 3), std_dp, dtype, device)
                + torch.randn(N_SENSORS, 3, dtype=dtype, device=device) * 1e-2 * self.k
            )
            dw = (
                self.walking_noise((T, N_SENSORS, 3), std_dw, dtype, device)
                + torch.randn(N_SENSORS, 3, dtype=dtype, device=device) * 1e-1 * self.k
            )

            dR = roma.rotvec_to_rotmat(dw.reshape(T * N_SENSORS, 3)).reshape(
                T, N_SENSORS, 3, 3
            )
            # Bone-to-sensor mount offset (GlobalPose RBS).
            rbs = self._random_rotation_matrix(N_SENSORS, dtype, device)
            position_imu = vertex + (grot @ dp.unsqueeze(-1)).squeeze(-1)
            rotation_imu = grot @ rbs.unsqueeze(0) @ dR
        else:
            position_imu = vertex
            rotation_imu = grot
            rbs = (
                torch.eye(3, dtype=dtype, device=device)
                .unsqueeze(0)
                .expand(N_SENSORS, 3, 3)
                .clone()
            )

        gw = self.gw.to(dtype=dtype, device=device)
        if acc_mode == "savgol":
            acc = get_local_acc_sav_gol(position_imu, rotation_imu, fps, gw)
        else:  # "paper": 2nd-order finite differences (GlobalPose replica)
            acc = get_local_acc_finite_diff(position_imu, rotation_imu, fps, gw)
        gyr = get_local_ang_vel_diff(rotation_imu, fps)

        if add_noise:
            import math as _math

            dt_scale = _math.sqrt(1.0 / fps)
            acc = (
                acc
                + torch.randn_like(acc) * _ACC_WHITE
                + self.walking_noise(acc.shape, _ACC_WALK * dt_scale, dtype, device)
            )
            gyr = (
                gyr
                + torch.randn_like(gyr) * _GYR_WHITE
                + self.walking_noise(gyr.shape, _GYR_WALK * dt_scale, dtype, device)
            )

        rot_sensor_world = integrate_ori_gyro(rotation_imu, gyr, fps)
        if add_noise:
            rot_sensor_world = add_ori_noise(rot_sensor_world, self.k)
        # Unmount to bone-in-world (GlobalPose R_sim @ RBS.T).
        rot_bone_world = unmount_to_bone(rot_sensor_world, rbs)

        rot_lab_world = (
            torch.eye(3, dtype=dtype, device=device)
            .unsqueeze(0)
            .expand(N_SENSORS, 3, 3)
            .clone()
        )
        rot_lab_calib = rot_lab_world.clone()

        joint_centered = all_joint - translation.unsqueeze(1)

        return {
            "FPS": fps,
            "RBW": rot_bone_world,
            "ACC": acc,
            "GYR": gyr,
            "RLW": rot_lab_world,
            "RLC": rot_lab_calib,
            "RSB": rbs,
            "Pose": pose,
            "Trans": translation,
            "Grot": all_grot_out,
            "Joint": joint_centered,
        }
