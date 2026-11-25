"""Spatial (Plücker) transforms and body Jacobian (Featherstone 2008, §4.2).

All spatial vectors follow the Featherstone convention:
    v = [ω]   ← angular (top 3)
        [v]   ← linear  (bottom 3)

Plücker coordinate transform from parent frame to body-i frame:
    X = [ E       0  ]
        [-E · r×  E  ]
where E is the 3×3 rotation (parent→child) and r is the translation
vector expressed in the parent frame.
"""

from __future__ import annotations

import torch

from .homogeneous import skewv

__all__ = [
    "plucker_transform",
    "inverse_plucker_transform",
    "motion_subspace",
    "ancestor_mask",
    "body_jacobian",
    "body_jacobian_autograd",
]

# ── Plücker transform primitives ──────────────────────────────────────────────


def plucker_transform(E: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Build the 6×6 Plücker spatial transform matrix.

    Implements  X = Xrot(E) · Xxlt(r)  from Featherstone Eq. 2.24.

    Args:
        E: (..., 3, 3) rotation matrix (parent frame → child frame).
        r: (..., 3) translation from parent origin to child origin,
           expressed in the **parent** frame.

    Returns:
        X: (..., 6, 6) Plücker transform matrix.
    """
    rx = skewv(r)  # (..., 3, 3)
    zeros = torch.zeros_like(E)
    top = torch.cat([E, zeros], dim=-1)  # (..., 3, 6)
    bot = torch.cat([-E @ rx, E], dim=-1)  # (..., 3, 6)
    return torch.cat([top, bot], dim=-2)  # (..., 6, 6)


def inverse_plucker_transform(X: torch.Tensor) -> torch.Tensor:
    """Compute the inverse of a Plücker spatial transform.

    For  X = [E, 0; -E·r×, E],  the inverse is:
        X⁻¹ = [Eᵀ, 0; r×·Eᵀ, Eᵀ]

    Implemented analytically (no torch.linalg.inv) so gradients flow
    through the rotation and translation components.

    Args:
        X: (..., 6, 6) Plücker transform.

    Returns:
        (..., 6, 6) inverse Plücker transform.
    """
    Et = X[..., :3, :3].transpose(-1, -2)  # (..., 3, 3)  E^T
    # From X[3:6, 0:3] = -E·r×  →  r× = -E^T · X[3:6, 0:3]
    rx = -Et @ X[..., 3:, :3]  # (..., 3, 3)  r× (in child frame)
    zeros = torch.zeros_like(Et)
    top = torch.cat([Et, zeros], dim=-1)  # (..., 3, 6)
    bot = torch.cat([rx @ Et, Et], dim=-1)  # (..., 3, 6)
    return torch.cat([top, bot], dim=-2)  # (..., 6, 6)


# ── Motion subspace matrices ───────────────────────────────────────────────────

_MOTION_SUBSPACE: dict[str, list[float]] = {
    "revolute_x": [1, 0, 0, 0, 0, 0],
    "revolute_y": [0, 1, 0, 0, 0, 0],
    "revolute_z": [0, 0, 1, 0, 0, 0],
    "prismatic_x": [0, 0, 0, 1, 0, 0],
    "prismatic_y": [0, 0, 0, 0, 1, 0],
    "prismatic_z": [0, 0, 0, 0, 0, 1],
}


def motion_subspace(
    joint_type: str,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the motion subspace matrix S for a given joint type.

    Spatial vector convention: [angular; linear].

    Args:
        joint_type: One of ``"revolute_x"``, ``"revolute_y"``,
                    ``"revolute_z"``, ``"prismatic_x"``, ``"prismatic_y"``,
                    ``"prismatic_z"``, or ``"ball"``.
        dtype:  Output dtype.
        device: Output device.

    Returns:
        S: (6, n_f) motion subspace matrix. n_f=1 for single-axis joints,
           n_f=3 for ``"ball"``.
    """
    if joint_type == "ball":
        # 3-DOF ball joint: angular axes x, y, z; no linear contribution
        S = torch.zeros(6, 3, dtype=dtype, device=device)
        S[0, 0] = 1.0
        S[1, 1] = 1.0
        S[2, 2] = 1.0
        return S
    if joint_type not in _MOTION_SUBSPACE:
        raise ValueError(
            f"Unknown joint type '{joint_type}'. "
            f"Valid types: {list(_MOTION_SUBSPACE) + ['ball']}"
        )
    col = torch.tensor(_MOTION_SUBSPACE[joint_type], dtype=dtype, device=device)
    return col.unsqueeze(-1)  # (6, 1)


# ── Ancestor mask ──────────────────────────────────────────────────────────────


def ancestor_mask(parents: list[int]) -> torch.Tensor:
    """Precompute the (J, J) ancestor relationship matrix.

    ``mask[i, j]`` is True if joint j is an ancestor of joint i
    (including i itself, i.e. j == i).

    Args:
        parents: Length-J list.  ``parents[0] = -1`` (root).
                 ``parents[i]`` is the parent index of joint i for i > 0.

    Returns:
        (J, J) bool tensor.
    """
    J = len(parents)
    mask = torch.zeros(J, J, dtype=torch.bool)
    for i in range(J):
        j = i
        while j >= 0:
            mask[i, j] = True
            j = parents[j] if j > 0 else -1
    return mask


# ── Body Jacobian ──────────────────────────────────────────────────────────────


def body_jacobian(
    E: torch.Tensor,
    r: torch.Tensor,
    parents: list[int],
    S: torch.Tensor,
) -> torch.Tensor:
    """Compute the body Jacobian for every link in a kinematic tree.

    Implements the body-coordinate recursive forward pass from
    Featherstone Table 5.1 to build the accumulated Plücker transforms,
    then assembles the full body Jacobian (Featherstone Eq. 4.6) in a
    single vectorised step.

    The j-th block of Jacobian columns for body i is:
        J_i[:, j*n_f:(j+1)*n_f] = (^i X_0) · (^j X_0)⁻¹ · S_j
    if j is an ancestor of i (including i itself), zero otherwise.

    Args:
        E:       (..., J, 3, 3) local rotation matrices (parent→child).
        r:       (..., J, 3) local translation vectors in the parent frame.
        parents: Length-J list. ``parents[0] = -1`` (root). ``parents[i]``
                 is the parent index of joint i for i > 0.
        S:       (J, 6, n_f) stacked motion subspace matrices.  Use
                 :func:`motion_subspace` to build each column and
                 ``torch.stack`` to combine.

    Returns:
        J: (..., J, 6, J*n_f) body Jacobian.  J[..., i, :, j*n_f:(j+1)*n_f]
           is the contribution of joint j to body i's spatial velocity.
    """
    dtype, device = E.dtype, E.device
    batch = E.shape[:-3]
    J_n = E.shape[-3]
    n_f = S.shape[-1]
    N = J_n * n_f

    S_b = S.to(dtype=dtype, device=device)  # (J, 6, n_f)

    # Pre-compute per-joint local Plücker transforms: ^i X_parent.
    X_local = [
        plucker_transform(E[..., i, :, :], r[..., i, :])  # (..., 6, 6)
        for i in range(J_n)
    ]

    # S padded to full column count: S_padded[i] has S[i] at column block i,
    # zeros elsewhere.  Shape (..., J, 6, N).
    S_padded = torch.zeros(J_n, 6, N, dtype=dtype, device=device)
    for i in range(J_n):
        S_padded[i, :, i * n_f : (i + 1) * n_f] = S_b[i]
    S_padded = S_padded.expand(*batch, J_n, 6, N)  # (..., J, 6, N)

    # ── Recursive forward pass (Featherstone Table 5.1) ───────────────────────
    # Spatial velocity recursion:  v_i = X_local_i @ v_parent + S_i @ dq_i
    # Differentiating:  J_i = X_local_i @ J_parent + S_i_padded
    # This is O(J) instead of the O(J^2) all-pairs formulation, since only
    # ancestor relationships contribute to the body Jacobian.
    J_list: list[torch.Tensor] = []
    # Root: no parent, only its own motion subspace at column block 0.
    J_list.append(S_padded[..., 0, :, :].clone())  # (..., 6, N)

    for i in range(1, J_n):
        j_parent = J_list[parents[i]]  # (..., 6, N)
        J_i = X_local[i] @ j_parent + S_padded[..., i, :, :]  # (..., 6, N)
        J_list.append(J_i)

    # Stack along the body axis: (..., J, 6, N) = (..., J, 6, J*n_f)
    return torch.stack(J_list, dim=-3)


def body_jacobian_autograd(
    E: torch.Tensor,
    r: torch.Tensor,
    parents: list[int],
    S: torch.Tensor,
) -> torch.Tensor:
    """Compute the body Jacobian via PyTorch autograd (reference implementation).

    Defines the spatial-velocity forward pass explicitly, then differentiates
    it with respect to a joint-velocity vector ``dq`` using
    :func:`torch.autograd.functional.jacobian`.  The result is mathematically
    identical to :func:`body_jacobian` but is computed by automatic
    differentiation rather than the analytical Plücker formula.

    This function is primarily intended as a **test oracle** to verify that
    :func:`body_jacobian` is correct.  It does not support batched inputs
    and should not be used in production (it is O(J·n_f) backward passes).

    Args:
        E:       (J, 3, 3) local rotation matrices — **no batch dim**.
        r:       (J, 3) local translation vectors — **no batch dim**.
        parents: Length-J list.  ``parents[0] = -1``.
        S:       (J, 6, n_f) motion subspace matrices.

    Returns:
        Jac: (J, 6, J*n_f) body Jacobian, computed via autograd.
    """
    J_n = E.shape[0]
    n_f = S.shape[-1]
    dtype, device = E.dtype, E.device
    S = S.to(dtype=dtype, device=device)

    def _spatial_velocity(dq_flat: torch.Tensor) -> torch.Tensor:
        """Forward pass: dq_flat (J*n_f,) → body spatial velocities (J, 6)."""
        vel: list[torch.Tensor] = [torch.zeros(6, dtype=dtype, device=device)] * J_n
        # Root: no parent transform
        vel[0] = S[0] @ dq_flat[:n_f]
        for i in range(1, J_n):
            X_i = plucker_transform(E[i], r[i])
            dq_i = dq_flat[i * n_f : (i + 1) * n_f]
            vel[i] = X_i @ vel[parents[i]] + S[i] @ dq_i
        return torch.stack(vel)  # (J, 6)

    dq_zero = torch.zeros(J_n * n_f, dtype=dtype, device=device)
    # jacobian output shape: (J, 6, J*n_f)
    Jac = torch.autograd.functional.jacobian(_spatial_velocity, dq_zero)
    return Jac  # type: ignore[return-value]
