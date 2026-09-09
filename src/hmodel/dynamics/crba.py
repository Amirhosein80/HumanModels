import torch

from ..core.jacobian import body_jacobian


def crba(
    E: torch.Tensor,
    r: torch.Tensor,
    parents: list[int],
    S: torch.Tensor,
    I_spatial: torch.Tensor,
) -> torch.Tensor:
    """Generalized inertia matrix M(q) via the Composite Rigid Body Algorithm.

    Computes M(q) = Σ_i  J_i(q)ᵀ · I_i · J_i(q)

    where J_i is the (6 × N) body Jacobian for body i and I_i is its 6×6
    spatial inertia.  This is the Jacobian-based equivalent of Featherstone's
    CRBA (§6.2) and is fully vectorised (no Python loop over body pairs).

    Args:
        E:         (J, 3, 3) local rotation matrices (parent → child).
        r:         (J, 3) local translation vectors in parent frame.
        parents:   Length-J list.  ``parents[0] = -1`` (root).
        S:         (J, 6, n_f) stacked motion subspace matrices.
        I_spatial: (J, 6, 6) spatial inertia for each body (in body frame).

    Returns:
        M: (N, N) symmetric positive-definite inertia matrix,  N = J * n_f.
    """
    Jac = body_jacobian(E, r, parents, S)  # (J, 6, N)
    F = torch.einsum("jab,jbc->jac", I_spatial, Jac)  # I_i @ J_i → (J, 6, N)
    return torch.einsum("jkn,jkm->nm", Jac, F)  # Σ_j Jᵀ F  → (N, N)
