import torch

from ..core.dynamics import force_cross, motion_cross
from ..core.jacobian import plucker_transform


def rnea(
    E: torch.Tensor,
    r: torch.Tensor,
    parents: list[int],
    S: torch.Tensor,
    I_spatial: torch.Tensor,
    dq: torch.Tensor,
    gravity: torch.Tensor,
) -> torch.Tensor:
    """Bias forces h(q, q̇) via the Recursive Newton-Euler Algorithm.

    Computes the generalised forces needed to produce zero joint acceleration
    at the given state (q, q̇), including gravity, Coriolis, and centripetal
    effects.  Gravity is handled as a fictitious base acceleration (Featherstone
    Table 5.1) so no external force vector is needed.

    Args:
        E:         (J, 3, 3) local rotation matrices (parent → child).
        r:         (J, 3) local translation vectors in parent frame.
        parents:   Length-J list.  ``parents[0] = -1`` (root).
        S:         (J, 6, n_f) stacked motion subspace matrices.
        I_spatial: (J, 6, 6) spatial inertia for each body (in body frame).
        dq:        (N,) generalised joint velocities,  N = J * n_f.
        gravity:   (3,) gravitational acceleration vector in world frame
                   (e.g. ``torch.tensor([0., -9.81, 0.])``).

    Returns:
        h: (N,) bias force vector h(q, q̇).
    """
    J_n = E.shape[0]
    n_f = S.shape[-1]
    dtype, device = E.dtype, E.device

    # Spatial gravity: [0; g] in world frame (= body-0 frame for root)
    a_grav = torch.zeros(6, dtype=dtype, device=device)
    a_grav[3:] = gravity.to(dtype=dtype, device=device)

    # Pre-compute local Plücker transforms X[i]: parent → body-i
    X_local = [plucker_transform(E[i], r[i]) for i in range(J_n)]

    # ── Forward pass (root → leaves) ─────────────────────────────────────────
    v: list[torch.Tensor] = []
    a: list[torch.Tensor] = []

    # Root: parent is the fixed world; world v=0, world a = −a_grav (fictitious)
    v0 = S[0] @ dq[:n_f]
    a0 = -a_grav + motion_cross(v0) @ (S[0] @ dq[:n_f])
    v.append(v0)
    a.append(a0)

    for i in range(1, J_n):
        Xi = X_local[i]
        dq_i = dq[i * n_f : (i + 1) * n_f]
        vi = Xi @ v[parents[i]] + S[i] @ dq_i
        ai = Xi @ a[parents[i]] + motion_cross(vi) @ (S[i] @ dq_i)
        v.append(vi)
        a.append(ai)

    # ── Backward pass (leaves → root) ────────────────────────────────────────
    f: list[torch.Tensor] = []
    for i in range(J_n):
        Ivi = I_spatial[i] @ v[i]
        fi = I_spatial[i] @ a[i] + force_cross(v[i]) @ Ivi
        f.append(fi)

    tau = torch.zeros(J_n * n_f, dtype=dtype, device=device)
    for i in range(J_n - 1, -1, -1):
        tau[i * n_f : (i + 1) * n_f] = S[i].T @ f[i]
        if parents[i] >= 0:
            # Propagate force to parent: X^T maps child force → parent force
            f[parents[i]] = f[parents[i]] + X_local[i].T @ f[i]

    return tau
