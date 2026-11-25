"""Rigid-body dynamics primitives (Featherstone 2008).

Provides the low-level building blocks used by the ``hmodel.dynamics``
algorithms (CRBA, RNEA, contact Jacobian):

- :func:`spatial_inertia` — 6×6 spatial inertia matrix from physical properties
- :func:`motion_cross`    — 6×6 motion cross-product matrix  v×
- :func:`force_cross`     — 6×6 force cross-product matrix   v×*

All functions follow the Featherstone spatial vector convention:
    v = [ω]   ← angular (top 3)
        [v]   ← linear  (bottom 3)
"""

from __future__ import annotations

import torch

from .homogeneous import skewv

__all__ = [
    "spatial_inertia",
    "motion_cross",
    "force_cross",
]


# ── Spatial inertia ───────────────────────────────────────────────────────────


def spatial_inertia(
    mass: torch.Tensor,
    com: torch.Tensor,
    inertia: torch.Tensor,
) -> torch.Tensor:
    """Build 6×6 Featherstone spatial inertia matrices.

    Implements Featherstone Eq. 2.63 for the [ω; v] convention:

        I = [ I_c − m·c×·c×,   m·c×  ]
            [     −m·c×,        m·I₃  ]

    where c× = skewv(c) and I_c is the inertia tensor at the CoM.

    Args:
        mass:    (..., J) per-body mass in kg.
        com:     (..., J, 3) CoM offset from body origin, expressed in body frame.
        inertia: (..., J, 3, 3) inertia tensor at the CoM, in body frame.

    Returns:
        (..., J, 6, 6) spatial inertia matrices.
    """
    dtype, device = com.dtype, com.device
    cx = skewv(com)  # (..., J, 3, 3)
    m = mass[..., None, None]  # (..., J, 1, 1)

    eye3 = torch.eye(3, dtype=dtype, device=device).expand(*cx.shape[:-2], 3, 3)

    top_left = inertia - m * (cx @ cx)  # I_c − m c×²
    top_right = m * cx  # m c×
    bot_left = -m * cx  # −m c×
    bot_right = m * eye3  # m I₃

    top = torch.cat([top_left, top_right], dim=-1)  # (..., J, 3, 6)
    bot = torch.cat([bot_left, bot_right], dim=-1)
    return torch.cat([top, bot], dim=-2)  # (..., J, 6, 6)


# ── Spatial cross-product operators ───────────────────────────────────────────


def motion_cross(v: torch.Tensor) -> torch.Tensor:
    """6×6 motion cross-product matrix for spatial velocity v.

    For v = [ω; v_lin]:

        v× = [ ω×,   0  ]
             [ v×,  ω×  ]

    Args:
        v: (..., 6) spatial velocity.

    Returns:
        (..., 6, 6) motion cross-product matrix.
    """
    omega = v[..., :3]
    vlin = v[..., 3:]
    ox = skewv(omega)
    vx = skewv(vlin)
    zeros = torch.zeros_like(ox)
    top = torch.cat([ox, zeros], dim=-1)
    bot = torch.cat([vx, ox], dim=-1)
    return torch.cat([top, bot], dim=-2)  # (..., 6, 6)


def force_cross(v: torch.Tensor) -> torch.Tensor:
    """6×6 force cross-product matrix  v×* = −(v×)ᵀ.

    Args:
        v: (..., 6) spatial velocity.

    Returns:
        (..., 6, 6) force cross-product matrix.
    """
    return -motion_cross(v).transpose(-1, -2)


# ── Matrix-free spatial cross-product applications ─────────────────────────


def motion_cross_apply(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Compute ``motion_cross(v) @ u`` without building the 6×6 matrix.

    For v = [ω; v_lin] and u = [u_ang; u_lin]:

        v× u = [ω × u_ang;
                v_lin × u_ang + ω × u_lin]

    Args:
        v: (..., 6) spatial velocity.
        u: (..., 6) spatial vector.

    Returns:
        (..., 6) result of v× u.
    """
    omega = v[..., :3]
    vlin = v[..., 3:]
    u_ang = u[..., :3]
    u_lin = u[..., 3:]
    top = torch.linalg.cross(omega, u_ang, dim=-1)
    bot = torch.linalg.cross(vlin, u_ang, dim=-1) + torch.linalg.cross(
        omega, u_lin, dim=-1
    )
    return torch.cat([top, bot], dim=-1)


def force_cross_apply(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Compute ``force_cross(v) @ f`` without building the 6×6 matrix.

    For v = [ω; v_lin] and f = [f_top; f_bot]:

        v×* f = [ω × f_top + v_lin × f_bot;
                 ω × f_bot]

    Args:
        v: (..., 6) spatial velocity.
        f: (..., 6) spatial force.

    Returns:
        (..., 6) result of v×* f.
    """
    omega = v[..., :3]
    vlin = v[..., 3:]
    f_top = f[..., :3]
    f_bot = f[..., 3:]
    top = torch.linalg.cross(omega, f_top, dim=-1) + torch.linalg.cross(
        vlin, f_bot, dim=-1
    )
    bot = torch.linalg.cross(omega, f_bot, dim=-1)
    return torch.cat([top, bot], dim=-1)
