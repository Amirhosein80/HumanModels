"""STGIPoser synthesis operators (ported).

Preference-free replicas of ``src/preprocess_utils.py`` from
``https://github.com/Amirhosein80/STGIPoser`` (MIT): the PIP-style
wide-stencil acceleration, root-frame velocity, and pairwise UWB distances.

Deltas from upstream: device/dtype are inherited from inputs (upstream
hardcodes ``"cuda"``) and ``fps`` is parameterised (upstream hardcodes 60 Hz,
i.e. ``3600 == 60**2``). At ``fps=60`` the numerics are identical.
"""

from __future__ import annotations

import torch

__all__ = [
    "V_MASK",
    "TC_IMU_MASK",
    "AMASS_ROT_ROWS",
    "IG_JOINTS",
    "syn_acc",
    "syn_vel",
    "syn_uwb",
]

# Sensor vertices: pelvis, L/R knee, head, L/R wrist (repo pelvis-first order).
V_MASK = [3021, 1176, 4662, 411, 1961, 5424]
# pkl channels are GlobalPose-ordered [L_LowArm, R_LowArm, L_LowLeg, R_LowLeg,
# Head, Pelvis]; repo slot i takes pkl channel TC_IMU_MASK[i].
TC_IMU_MASK = [5, 2, 3, 4, 0, 1]
# AMASS global-frame alignment rows (upstream ``AMASS_ROT``).
AMASS_ROT_ROWS = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
# Ignored joints, zeroed before FK (matches upstream ``IG_JOINTS``).
IG_JOINTS = [7, 8, 10, 11, 20, 21, 22, 23]


def amass_rot(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """AMASS global-frame alignment matrix, shape (1, 3, 3)."""
    return torch.tensor(AMASS_ROT_ROWS, dtype=dtype, device=device).unsqueeze(0)


def syn_acc(trans: torch.Tensor, fps: float = 60.0) -> torch.Tensor:
    """Wide-stencil acceleration from positions (upstream ``_syn_acc`` replica).

    Narrow ``[t-1, t, t+1]`` stencil everywhere except the borders, then the
    interior ``[4:-4]`` is overwritten with the wide ``[t-4, t, t+4]`` stencil
    scaled by ``1/16``. Border frames stay zero.

    Args:
        trans: (..., T, S, 3) positions, time axis must be dim -3 with T >= 9.
        fps:   Sampling rate in Hz (upstream hardcodes 60).

    Returns:
        acc: Same shape as ``trans`` (world-frame, no gravity term).
    """
    if trans.shape[-3] < 9:
        raise ValueError(
            "syn_acc needs T >= 9 frames for the wide stencil, "
            f"got {trans.shape[-3]}"
        )
    scale = fps * fps
    acc = torch.zeros_like(trans)
    acc[..., 1:-1, :, :] = (
        trans[..., 0:-2, :, :] + trans[..., 2:, :, :] - 2 * trans[..., 1:-1, :, :]
    ) * scale
    acc[..., 4:-4, :, :] = (
        trans[..., 0:-8, :, :] + trans[..., 8:, :, :] - 2 * trans[..., 4:-4, :, :]
    ) * (scale / 16)
    return acc


def syn_vel(trans: torch.Tensor, root: torch.Tensor, fps: float = 60.0) -> torch.Tensor:
    """Root-frame velocity from positions (upstream ``_syn_vel`` replica).

    Args:
        trans: (N, J, 3) joint positions.
        root:  (N, 1, 3, 3) root rotation matrices.
        fps:   Sampling rate in Hz (upstream hardcodes 60).

    Returns:
        vel: (N, J, 3) velocity in the root frame.
    """
    vel = torch.zeros_like(trans)
    vel[1:] = trans[1:] - trans[:-1]
    vel = root.transpose(-1, -2) @ vel.unsqueeze(-1)
    return vel[..., 0] * fps


def syn_uwb(p: torch.Tensor) -> torch.Tensor:
    """Pairwise UWB distances (upstream ``_syn_uwb`` replica).

    Args:
        p: (..., S, 3) sensor/vertex positions.

    Returns:
        d: (..., S, S) pairwise distances.
    """
    return torch.cdist(p, p)
