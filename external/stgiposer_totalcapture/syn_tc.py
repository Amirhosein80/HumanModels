"""TotalCapture synthesis (adapted port of STGIPoser ``process_total_capture``).

Upstream: ``src/preprocess.py::process_total_capture`` from
``https://github.com/Amirhosein80/STGIPoser`` (MIT), itself derived from
PIP/DynaIP preprocessing.

Pipeline per sequence: load the ``TotalCapture_Real_60FPS`` pkl (real
``acc``/``ori`` in GlobalPose channel order) + the AMASS pose npz, remap
channels via ``TC_IMU_MASK``, step the motion to the IMU rate, fix the root
frame with ``AMASS_ROT``, run FK, synthesise vertex acceleration with the
wide-stencil ``syn_acc``, **bias-match the real accelerometer to the
synthetic mean** (``acc = acc + (vacc.mean - acc.mean)`` — this is how
upstream reconciles the gravity/bias gap; ``imu_ori`` stays real), then
``syn_vel``/``syn_uwb``. Output schema matches upstream npz keys so the
existing ``extract_data_seq`` flow keeps working.

Deltas from upstream (see README.md):
- Mean SMPL shape (HModel ``SMPL`` takes no ``betas``); subject shape is a
  known, reported caveat.
- Device/dtype-agnostic; no hardcoded ``"cuda"``.
- Explicit ``pkl_path``/``pose_npz_path`` arguments instead of hardcoded
  ``"TotalCapture_Real_60FPS"`` path surgery.
- ``fps`` parameterised (upstream hardcodes 60 Hz ``mu``).
"""

from __future__ import annotations

import glob
import os
import pickle

import numpy as np
import roma
import torch

from .synth_ops import (
    IG_JOINTS,
    TC_IMU_MASK,
    V_MASK,
    amass_rot,
    syn_acc,
    syn_uwb,
    syn_vel,
)

__all__ = [
    "step_for_framerate",
    "prepare_motion",
    "synthesize_tc_sequence",
    "process_file",
    "run_folder",
    "save_npz",
]

#: Output keys, matching the upstream npz schema consumed by ``extract_data_seq``.
TARGET_KEYS = ("imu_acc", "imu_ori", "pose", "trans", "grot", "jvel", "uwb")

_N_JOINTS_AMASS = 52
_N_JOINTS_SMPL = 24


def step_for_framerate(framerate: float) -> int:
    """Downsampling step for a mocap framerate (upstream map).

    Args:
        framerate: E.g. 60, 100, 120, 150, 250.

    Returns:
        Integer step so the motion lands at ~60 Hz.
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
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Step AMASS motion to IMU rate and fix the root frame (upstream replica).

    Copies joint 37 over joint 23, keeps the first 24 joints, rotates the
    root joint and the translation with ``AMASS_ROT``, and makes translation
    root-relative.

    Args:
        poses_52:  (T_full, 52, 3) axis-angle poses.
        trans:     (T_full, 3) root translation.
        framerate: Mocap framerate for the step map.

    Returns:
        pose_24: (T, 24, 3) stepped axis-angle poses.
        trans:   (T, 3) stepped, root-relative, frame-fixed translation.
        step:    Downsampling step used.
    """
    step = step_for_framerate(framerate)
    dtype, device = poses_52.dtype, poses_52.device
    poses_52 = poses_52[::step].clone()
    trans = trans[::step].clone()
    poses_52[:, 23] = poses_52[:, 37]
    pose_24 = poses_52[:, :_N_JOINTS_SMPL].clone()
    rot = amass_rot(dtype, device)
    pose_24[:, 0] = roma.rotmat_to_rotvec(
        rot.matmul(roma.rotvec_to_rotmat(pose_24[:, 0]))
    )
    trans = trans - trans[0:1]
    trans = rot.matmul(trans.unsqueeze(-1)).view_as(trans)
    return pose_24, trans, step


def synthesize_tc_sequence(
    acc_real: torch.Tensor,
    ori_real: torch.Tensor,
    poses_52: torch.Tensor,
    trans_full: torch.Tensor,
    framerate: float,
    smpl,
    fps: float = 60.0,
) -> dict[str, torch.Tensor]:
    """Synthesise one TotalCapture sequence (upstream ``process_total_capture``).

    Args:
        acc_real:  (T_imu, 6, 3) real accelerometer (GlobalPose channel order).
        ori_real:  (T_imu, 6, 3, 3) real orientations (GlobalPose channel order).
        poses_52:  (T_mocap, 52, 3) AMASS axis-angle poses.
        trans_full:(T_mocap, 3) AMASS root translation.
        framerate: Mocap framerate.
        smpl:      Loaded :class:`~hmodel.bmodels.smpl.SMPL` (gender selected
                   by the caller; mean shape — see module docstring).
        fps:       IMU sampling rate in Hz.

    Returns:
        dict with upstream keys: ``imu_acc`` (bias-matched real acc),
        ``imu_ori`` (real ori), ``pose``, ``trans``, ``grot``, ``jvel``,
        ``uwb`` — plus ``imu_acc_raw`` (pre-match), ``vacc`` (synthetic
        vertex acceleration) and ``acc_bias_shift``.
    """
    acc = acc_real[:, TC_IMU_MASK].to(dtype=acc_real.dtype)
    ori = ori_real[:, TC_IMU_MASK]
    pose_24, trans, _ = prepare_motion(poses_52, trans_full, framerate)

    if acc.shape[0] < trans.shape[0]:
        trans = trans[: acc.shape[0]]
        pose_24 = pose_24[: acc.shape[0]]
    elif acc.shape[0] > pose_24.shape[0]:
        acc = acc[: pose_24.shape[0]]
        ori = ori[: pose_24.shape[0]]
    if trans.shape[0] != acc.shape[0]:
        raise ValueError(
            "length mismatch after trimming: "
            f"motion={trans.shape[0]}, imu={acc.shape[0]}"
        )

    rot_mat = roma.rotvec_to_rotmat(pose_24).view(-1, _N_JOINTS_SMPL, 3, 3)
    eye = torch.eye(3, dtype=rot_mat.dtype, device=rot_mat.device)
    rot_mat[:, IG_JOINTS] = eye

    grot, joint, vert = smpl.forward_kinematics(
        pose=rot_mat, root_position=trans, return_mesh=True
    )

    vacc = syn_acc(vert[:, V_MASK], fps=fps)
    bias_shift = vacc.mean(dim=0, keepdim=True) - acc.mean(dim=0, keepdim=True)
    acc_matched = acc + bias_shift

    jvel = syn_vel(joint, grot[:, 0:1], fps=fps)
    uwb = syn_uwb(vert[:, V_MASK])

    return {
        "imu_acc": acc_matched,
        "imu_acc_raw": acc,
        "vacc": vacc,
        "acc_bias_shift": bias_shift.squeeze(0),
        "imu_ori": ori,
        "pose": rot_mat,
        "trans": trans,
        "grot": grot,
        "jvel": jvel,
        "uwb": uwb,
    }


def _load_pkl(pkl_path: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return {
        "acc": torch.from_numpy(np.asarray(data["acc"])).to(dtype),
        "ori": torch.from_numpy(np.asarray(data["ori"])).to(dtype),
    }


def _load_npz(pose_npz_path: str, dtype: torch.dtype) -> dict:
    cdata = np.load(pose_npz_path)
    return {
        "poses_52": torch.from_numpy(np.asarray(cdata["poses"], dtype=np.float64))
        .to(dtype)
        .reshape(-1, _N_JOINTS_AMASS, 3),
        "trans": torch.from_numpy(np.asarray(cdata["trans"], dtype=np.float64)).to(
            dtype
        ),
        "framerate": float(cdata["mocap_framerate"]),
        "gender": str(cdata["gender"]),
    }


def process_file(
    pkl_path: str,
    pose_npz_path: str,
    smpl,
    fps: float = 60.0,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Process one pkl + pose-npz pair (upstream per-file body).

    Args:
        pkl_path:      ``TotalCapture_Real_60FPS/<s>_<action>.pkl``.
        pose_npz_path: Matching AMASS ``<action>_poses.npz``.
        smpl:          Loaded SMPL for the sequence gender.
        fps:           IMU sampling rate in Hz.
        dtype:         Compute dtype (upstream uses float32).

    Returns:
        Target dict from :func:`synthesize_tc_sequence`.
    """
    pkl = _load_pkl(pkl_path, dtype)
    npz = _load_npz(pose_npz_path, dtype)
    return synthesize_tc_sequence(
        acc_real=pkl["acc"],
        ori_real=pkl["ori"],
        poses_52=npz["poses_52"],
        trans_full=npz["trans"],
        framerate=npz["framerate"],
        smpl=smpl,
        fps=fps,
    )


def save_npz(path: str, targets: dict[str, torch.Tensor]) -> None:
    """Save upstream-schema npz (torch → numpy)."""
    arrays = {k: targets[k].detach().cpu().numpy() for k in TARGET_KEYS if k in targets}
    np.savez(path, **arrays)


def _pose_path_for(pkl_path: str, pose_root: str) -> tuple[str, str]:
    stem = os.path.splitext(os.path.basename(pkl_path))[0]  # e.g. s1_walking1
    subject, action = stem.split("_")[0].lower(), stem.split("_")[1]
    return os.path.join(pose_root, subject, f"{action}_poses.npz"), stem


def run_folder(
    pkl_root: str,
    pose_root: str,
    out_dir: str,
    smpl_by_gender: dict[str, object],
    fps: float = 60.0,
    dtype: torch.dtype = torch.float32,
) -> dict[str, list[str]]:
    """Batch driver over ``**/*.pkl`` (upstream folder loop).

    Args:
        pkl_root:       Directory with ``TotalCapture_Real_60FPS``-style pkls.
        pose_root:      Directory with ``<subject>/<action>_poses.npz`` files.
        out_dir:        Output directory (mirrors ``<subject>/`` subfolders).
        smpl_by_gender: ``{"male": SMPL, "female": SMPL}`` instances.
        fps:           IMU sampling rate in Hz.
        dtype:         Compute dtype.

    Returns:
        ``{"done": [...], "skipped": [...]}`` with sequence stems / reasons.
    """
    done, skipped = [], []
    for pkl_path in sorted(glob.glob(os.path.join(pkl_root, "**", "*.pkl"))):
        pose_path, stem = _pose_path_for(pkl_path, pose_root)
        if not os.path.exists(pose_path):
            skipped.append(f"{stem}: missing pose npz")
            continue
        gender = str(np.load(pose_path)["gender"])
        try:
            targets = process_file(
                pkl_path, pose_path, smpl_by_gender[gender], fps=fps, dtype=dtype
            )
        except ValueError as exc:  # e.g. unsupported framerate
            skipped.append(f"{stem}: {exc}")
            continue
        subject = stem.split("_")[0].lower()
        os.makedirs(os.path.join(out_dir, subject), exist_ok=True)
        save_npz(os.path.join(out_dir, subject, stem + ".npz"), targets)
        done.append(stem)
    return {"done": done, "skipped": skipped}
