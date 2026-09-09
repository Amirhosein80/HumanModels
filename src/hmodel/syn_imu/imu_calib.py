"""IMU calibration: transform raw IMU measurements between reference frames."""

from __future__ import annotations

import torch

__all__ = ["imu_calibration"]

# Gravity in world/lab frame — matches GlobalPose gW = (0, -9.8, 0).
_GRAVITY = (0.0, -9.8, 0.0)


def imu_calibration(
    acc: torch.Tensor,
    gyro: torch.Tensor,
    rot_bone_world: torch.Tensor,
    rot_lab_world: torch.Tensor,
    rot_lab_calib: torch.Tensor,
    remove_g: bool = True,
    rot_sensor_bone: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Transform raw IMU measurements into the lab/world frame.

    Follows the GlobalPose convention (``test.py::compare_realimu``):
    ``aM = RIM^T @ RIS @ aS + g``, ``wM = RIM^T @ RIS @ wS`` — both acc and
    gyro land in the same lab/world frame. The bone-to-sensor mount
    (``SynIMU`` ``RSB`` output) is applied first so sensor-local readings
    are correctly unmounted before the bone-to-lab rotation.

    Args:
        acc:            (T, S, 3) raw accelerometer readings in sensor local frame.
        gyro:           (T, S, 3) raw gyroscope readings in sensor local frame.
        rot_bone_world: (T, S, 3, 3) bone orientations in world frame.
        rot_lab_world:  (S, 3, 3) lab-to-world rotation (identity when lab == world).
        rot_lab_calib:  (S, 3, 3) lab-to-calibrated-frame rotation.
        remove_g:       If True, add the gravity offset ``[0, -9.8, 0]`` to the
                        calibrated accelerometer (lab frame).
        rot_sensor_bone: (S, 3, 3) bone-to-sensor mount (``RSB`` from
                        :meth:`SynIMU.syn_imu`). Defaults to identity for
                        back-compat when no mount was modelled.

    Returns:
        dict with keys:
            ``acc_calib``       (T, S, 3) lab-frame accelerometer
            ``gyro_calib``      (T, S, 3) lab-frame gyroscope
            ``rot_bone_calib``  (T, S, 3, 3) bone orientation in calib frame
    """
    if rot_sensor_bone is None:
        rot_sensor_bone = (
            torch.eye(3, dtype=acc.dtype, device=acc.device)
            .unsqueeze(0)
            .expand(rot_lab_world.shape[0], 3, 3)
        )
    # Bone rotation in lab frame: R_bone_lab = R_lab_world^T @ R_bone_world
    rot_bone_lab = rot_lab_world.transpose(-1, -2) @ rot_bone_world
    rot_bone_calib = rot_bone_lab @ rot_lab_calib

    # Sensor orientation in lab frame: R_sens_lab = R_bone_lab @ R_sens_bone
    rot_sensor_lab = rot_bone_lab @ rot_sensor_bone.unsqueeze(0)

    # Rotate both modalities into the lab frame (same frame).
    gyro_calib = (rot_sensor_lab @ gyro.unsqueeze(-1)).squeeze(-1)
    acc_calib = (rot_sensor_lab @ acc.unsqueeze(-1)).squeeze(-1)
    if remove_g:
        gw = torch.tensor(_GRAVITY, dtype=acc.dtype, device=acc.device)
        g_lab = (rot_lab_world.transpose(-1, -2) @ gw.unsqueeze(-1)).squeeze(-1)
        acc_calib = acc_calib + g_lab.unsqueeze(0)

    return {
        "acc_calib": acc_calib,
        "gyro_calib": gyro_calib,
        "rot_bone_calib": rot_bone_calib,
    }
