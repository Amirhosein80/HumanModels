"""TotalCapture real-data loading and calibration.

Utilities to bring raw TotalCapture Xsens recordings into the same frame as
synthetic IMU from :class:`~hmodel.syn_imu.SynIMU`, following the GlobalPose
conventions (``process.py`` / ``test.py::compare_realimu``):

- :func:`parse_sensors` reads ``*_Xsens_AuxFields.sensors`` files (quaternion
  ``wxyz`` + acc/gyro/mag per line).
- :func:`load_calib` reads ``*_calib_imu_{bone,ref}.txt`` files (quaternion
  ``xyzw``) and applies the SMPL-frame fixes.
- :func:`prepare_motion` steps raw AMASS poses to IMU rate and applies the
  ``AMASS_ROT`` root fix — **required** before driving FK: raw npz poses live
  one global rotation away from the SMPL/world frame.
- :func:`calibrate` maps raw sensor readings to bone-in-calibrated-frame
  readings: ``RMB = RIM^T @ RIS @ RSB``, ``aM = RIM^T @ RIS @ aS + g``,
  ``wM = RIM^T @ RIS @ wS``.

Comparing syn output against *raw* ``RIS``/``aS`` is a frame error, not a
sensor error: raw readings still carry the bone-to-sensor mount and the
Xsens world frame, while syn ``RBW``/``ACC`` live in the bone/SMPL frame.
"""

from __future__ import annotations

import numpy as np
import roma
import torch

__all__ = [
    "BONE_FIX_ROWS",
    "FILE_IDX",
    "GRAVITY",
    "REF_FIX_ROWS",
    "REPO_JOINT_MASK",
    "SENSOR_NAMES",
    "bone_fix",
    "calibrate",
    "load_amass_npz",
    "load_calib",
    "orientation_error_deg",
    "parse_sensors",
    "prepare_motion",
    "ref_fix",
    "select_sensors",
    "stack_calib",
    "step_for_framerate",
]

#: Repo canonical sensor order (pelvis-first).
SENSOR_NAMES = ["Pelvis", "L_LowLeg", "R_LowLeg", "Head", "L_LowArm", "R_LowArm"]
#: Repo order → TotalCapture file order [Head, Sternum, Pelvis, L_UpArm,
#: R_UpArm, L_LowArm, R_LowArm, L_UpLeg, R_UpLeg, L_LowLeg, R_LowLeg, ...].
FILE_IDX = [2, 9, 10, 0, 5, 6]
#: Repo joints matching SENSOR_NAMES order.
REPO_JOINT_MASK = [0, 4, 5, 15, 18, 19]
#: Gravity in world frame, matching GlobalPose ``gW``.
GRAVITY = (0.0, -9.8, 0.0)
#: SMPL-frame fixes from GlobalPose ``process.py``.
BONE_FIX_ROWS = ((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0))
REF_FIX_ROWS = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))

_N_JOINTS_AMASS = 52
_N_JOINTS_SMPL = 24


def bone_fix(
    dtype: torch.dtype = torch.float64, device: torch.device | None = None
) -> torch.Tensor:
    """Bone-frame fix matrix, shape (3, 3)."""
    return torch.tensor(BONE_FIX_ROWS, dtype=dtype, device=device)


def ref_fix(
    dtype: torch.dtype = torch.float64, device: torch.device | None = None
) -> torch.Tensor:
    """Reference-frame fix matrix, shape (3, 3)."""
    return torch.tensor(REF_FIX_ROWS, dtype=dtype, device=device)


def parse_sensors(path: str, dtype: torch.dtype = torch.float64) -> dict[str, object]:
    """Parse a TotalCapture ``*_Xsens_AuxFields.sensors`` file.

    Column layout per sensor line: ``name, quat wxyz (4), acc (3), gyro (3),
    mag (3)`` — same semantics as GlobalPose ``process.py``.

    Args:
        path:  File path.
        dtype: Output dtype (device is always CPU; files are text).

    Returns:
        dict with ``names`` (list of file-order sensor names), ``RIS``
        (T, S, 3, 3) sensor orientations in world, ``aS``/``wS``/``mS``
        (T, S, 3) sensor-local readings.
    """
    with open(path, "r") as f:
        lines = f.read().splitlines()
    if not lines:
        raise ValueError(f"empty sensors file: {path!r}")
    try:
        n_sensors, n_frames = (int(v) for v in lines[0].split())
    except ValueError as exc:
        raise ValueError(f"bad header line in {path!r}: {lines[0]!r}") from exc
    names: list[str] = []
    quat, acc, gyr, mag = [], [], [], []
    pos = 1
    for frame in range(n_frames):
        if pos >= len(lines) or int(lines[pos]) != frame + 1:
            raise ValueError(f"frame index mismatch at frame {frame + 1} in {path!r}")
        pos += 1
        for _ in range(n_sensors):
            if pos >= len(lines):
                raise ValueError(f"truncated file {path!r} at frame {frame + 1}")
            parts = lines[pos].split()
            pos += 1
            if len(parts) != 14:
                raise ValueError(f"bad sensor line in {path!r}: {lines[pos - 1]!r}")
            values = [float(x) for x in parts[1:]]
            # File order is wxyz; roma uses xyzw.
            quat.append([values[1], values[2], values[3], values[0]])
            acc.append(values[4:7])
            gyr.append(values[7:10])
            mag.append(values[10:13])
            if len(names) < n_sensors:
                names.append(parts[0])
    quat_t = torch.tensor(quat, dtype=dtype).reshape(n_frames, n_sensors, 4)
    return {
        "names": names,
        "RIS": roma.unitquat_to_rotmat(quat_t.reshape(-1, 4)).reshape(
            n_frames, n_sensors, 3, 3
        ),
        "aS": torch.tensor(acc, dtype=dtype).reshape(n_frames, n_sensors, 3),
        "wS": torch.tensor(gyr, dtype=dtype).reshape(n_frames, n_sensors, 3),
        "mS": torch.tensor(mag, dtype=dtype).reshape(n_frames, n_sensors, 3),
    }


def load_calib(path: str, fix: torch.Tensor) -> dict[str, torch.Tensor]:
    """Load a ``*_calib_imu_{bone,ref}.txt`` file (file quats are xyzw).

    Args:
        path: File path (first line is the sensor count, then one
              ``name x y z w`` line per sensor).
        fix:  (3, 3) SMPL-frame fix (see :func:`bone_fix`/:func:`ref_fix`).

    Returns:
        dict mapping sensor name → (3, 3) rotation (already ``@ fix``).
    """
    with open(path, "r") as f:
        lines = f.read().splitlines()
    out: dict[str, torch.Tensor] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"bad calib line in {path!r}: {line!r}")
        x, y, z, w = (float(v) for v in parts[1:5])
        # File order is already xyzw (roma convention); the transpose matches
        # GlobalPose process.py (file quats encode the inverse direction).
        quat = torch.tensor([x, y, z, w], dtype=fix.dtype, device=fix.device)
        out[parts[0]] = roma.unitquat_to_rotmat(quat).t() @ fix
    if not out:
        raise ValueError(f"no calibration entries in {path!r}")
    return out


def stack_calib(
    calib: dict[str, torch.Tensor], order: list[str] | tuple[str, ...]
) -> torch.Tensor:
    """Stack per-sensor calib rotations in the given name order.

    Args:
        calib: Mapping from :func:`load_calib`.
        order: Sensor names, e.g. :data:`SENSOR_NAMES`.

    Returns:
        R: (S, 3, 3) stacked rotations.
    """
    try:
        return torch.stack([calib[name] for name in order])
    except KeyError as exc:
        raise ValueError(f"calibration missing sensor {exc}") from exc


def select_sensors(x: torch.Tensor, idx: list[int] | tuple[int, ...]) -> torch.Tensor:
    """Reorder the sensor axis (dim 1) of batched (T, S, ...) tensors.

    Args:
        x:   (T, S, ...) tensor in file order.
        idx: Sensor indices, e.g. :data:`FILE_IDX`.

    Returns:
        Tensor with sensors in the requested order.
    """
    t = torch.tensor(list(idx), dtype=torch.long, device=x.device)
    return torch.index_select(x, 1, t)


def load_amass_npz(path: str, dtype: torch.dtype = torch.float64) -> dict[str, object]:
    """Load an AMASS-fit pose npz (``poses``/``trans``/``mocap_framerate``).

    Args:
        path:  ``<action>_poses.npz`` path.
        dtype: Output dtype.

    Returns:
        dict with ``poses_52`` (T, 52, 3) axis-angle, ``trans`` (T, 3),
        ``framerate`` (float) and ``gender`` (str).
    """
    data = np.load(path)
    return {
        "poses_52": torch.from_numpy(np.asarray(data["poses"], dtype=np.float64))
        .to(dtype)
        .reshape(-1, _N_JOINTS_AMASS, 3),
        "trans": torch.from_numpy(np.asarray(data["trans"], dtype=np.float64)).to(
            dtype
        ),
        "framerate": float(data["mocap_framerate"]),
        "gender": str(data["gender"]),
    }


def step_for_framerate(framerate: float) -> int:
    """Downsampling step for a mocap framerate (lands at ~60 Hz).

    Args:
        framerate: E.g. 60, 100, 120, 150, 250.

    Returns:
        Integer step.
    """
    rate = int(framerate)
    if rate in (120, 100):
        return 2
    if rate in (60, 59, 50):
        return 1
    if rate == 250:
        return 5
    if rate == 150:
        return 3
    raise ValueError(f"framerate {framerate!r} not supported (need ~60 Hz source)")


def prepare_motion(
    poses_52: torch.Tensor,
    trans: torch.Tensor,
    framerate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Step AMASS motion to IMU rate and fix the root frame.

    Copies joint 37 over joint 23, keeps the first 24 joints, rotates the
    root joint and the translation with ``AMASS_ROT``, and makes the
    translation root-relative. Raw npz poses live one global rotation away
    from the SMPL/world frame — skipping this leaves a uniform ~90° error
    on every sensor.

    Args:
        poses_52:  (T_full, 52, 3) axis-angle poses.
        trans:     (T_full, 3) root translation.
        framerate: Mocap framerate for the step map.

    Returns:
        pose_24: (T, 24, 3) stepped axis-angle poses in the SMPL frame.
        trans:   (T, 3) stepped, root-relative, frame-fixed translation.
    """
    if poses_52.shape[-2] != _N_JOINTS_AMASS:
        raise ValueError(f"expected {_N_JOINTS_AMASS} joints, got {poses_52.shape[-2]}")
    step = step_for_framerate(framerate)
    dtype, device = poses_52.dtype, poses_52.device
    poses_52 = poses_52[::step].clone()
    trans = trans[::step].clone()
    poses_52[:, 23] = poses_52[:, 37]
    pose_24 = poses_52[:, :_N_JOINTS_SMPL].clone()
    rot = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        dtype=dtype,
        device=device,
    ).unsqueeze(0)
    pose_24[:, 0] = roma.rotmat_to_rotvec(
        rot.matmul(roma.rotvec_to_rotmat(pose_24[:, 0]))
    )
    trans = trans - trans[0:1]
    trans = rot.matmul(trans.unsqueeze(-1)).view_as(trans)
    return pose_24, trans


def calibrate(
    acc: torch.Tensor,
    gyro: torch.Tensor,
    rot_sensor_world: torch.Tensor,
    rot_lab_world: torch.Tensor,
    rot_sensor_bone: torch.Tensor,
    gravity: tuple[float, float, float] = GRAVITY,
) -> dict[str, torch.Tensor]:
    """Map raw IMU readings to the bone-in-calibrated frame.

    GlobalPose ``test.py::compare_realimu`` convention: both modalities land
    in the same lab/world frame and gravity is added back to the
    accelerometer ::

        RMB = RIM^T @ RIS @ RSB
        aM  = RIM^T @ RIS @ aS + g
        wM  = RIM^T @ RIS @ wS

    Args:
        acc:              (T, S, 3) raw sensor-local accelerometer.
        gyro:             (T, S, 3) raw sensor-local gyroscope.
        rot_sensor_world: (T, S, 3, 3) sensor orientations in world (``RIS``).
        rot_lab_world:    (S, 3, 3) lab-to-world rotation (``RIM``).
        rot_sensor_bone:  (S, 3, 3) bone-to-sensor mount (``RSB``).
        gravity:          World-frame gravity vector added to ``aM``.

    Returns:
        dict with ``rot_bone`` (T, S, 3, 3), ``acc`` (T, S, 3) and
        ``gyro`` (T, S, 3) in the calibrated frame.
    """
    rot_lab_world = rot_lab_world.to(dtype=acc.dtype, device=acc.device)
    rot_sensor_bone = rot_sensor_bone.to(dtype=acc.dtype, device=acc.device)
    rim_t = rot_lab_world.transpose(-1, -2).unsqueeze(0)
    rot_bone = rim_t @ rot_sensor_world @ rot_sensor_bone.unsqueeze(0)
    rot_world_lab = rim_t @ rot_sensor_world
    gyro_out = (rot_world_lab @ gyro.unsqueeze(-1)).squeeze(-1)
    acc_out = (rot_world_lab @ acc.unsqueeze(-1)).squeeze(-1)
    gw = torch.tensor(gravity, dtype=acc.dtype, device=acc.device)
    acc_out = acc_out + gw.unsqueeze(0).unsqueeze(0)
    return {"rot_bone": rot_bone, "acc": acc_out, "gyro": gyro_out}


def orientation_error_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between rotation-matrix pairs (avoids rotvec wrapping).

    Args:
        a: (..., 3, 3) rotation matrices.
        b: (..., 3, 3) rotation matrices (broadcastable against ``a``).

    Returns:
        angles: (...) degrees in [0, 180].
    """
    rot = a.transpose(-1, -2) @ b
    trace = rot.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(-1.0, 3.0)
    return torch.rad2deg(torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0)))
