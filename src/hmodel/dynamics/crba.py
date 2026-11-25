"""Composite Rigid Body Algorithm for the generalized inertia matrix.

Implements Featherstone's CRBA (Rigid Body Dynamics Algorithms, Alg. 6.2)
which computes M(q) in O(J^2) using the kinematic tree structure, avoiding
the O(J^3) cost of building and contracting full body Jacobians.

Algorithm:
  1. Compute global transforms ^i X_0 for every body (top-down).
  2. Compute B_i = sum_{m in subtree(i)} ^i X_m^T I_m ^i X_m, the composite
     rigid-body inertia of each subtree referred to body i's frame, bottom-up:
         B_i = I_i + sum_{c in children(i)} X^{c→i}^T B_c X^{c→i}
  3. Assemble M:
     M[j,k] = S_j^T ^k X_j^T B_k S_k   if j is an ancestor of k
              M[k,j]                   if k is an ancestor of j (symmetry)
              0                        otherwise

All spatial vectors follow the Featherstone [angular; linear] convention.
"""

from __future__ import annotations

import torch

from ..core.jacobian import (
    inverse_plucker_transform,
    plucker_transform,
)


def _build_children(parents: list[int]) -> list[list[int]]:
    """Return ``children[i]`` = list of child joint indices of body i."""
    children: list[list[int]] = [[] for _ in parents]
    for i in range(1, len(parents)):
        children[parents[i]].append(i)
    return children


def _is_ancestor_table(parents: list[int]) -> list[list[bool]]:
    """``ancestor[j][k]`` is True if joint j is an ancestor of joint k."""
    J_n = len(parents)
    anc = [[False] * J_n for _ in range(J_n)]
    for k in range(J_n):
        x = k
        while x >= 0:
            anc[x][k] = True  # x is an ancestor of k
            x = parents[x] if x > 0 else -1
    return anc


def crba(
    E: torch.Tensor,
    r: torch.Tensor,
    parents: list[int],
    S: torch.Tensor,
    I_spatial: torch.Tensor,
) -> torch.Tensor:
    """Generalized inertia matrix M(q) via the recursive Composite Rigid Body Algorithm.

    Computes M(q) using Featherstone's CRBA (§6.2), which exploits the
    kinematic tree to avoid constructing the full body Jacobian.  The result
    is identical to the Jacobian-based form M(q) = Σ_i J_iᵀ I_i J_i.

    Args:
        E:         (J, 3, 3) local rotation matrices (parent → child).
        r:         (J, 3) local translation vectors in parent frame.
        parents:   Length-J list. ``parents[0] = -1`` (root).
                   ``parents[i]`` is the parent index of joint i for i > 0.
        S:         (J, 6, n_f) stacked motion subspace matrices.
        I_spatial: (J, 6, 6) spatial inertia for each body (in body frame).

    Returns:
        M: (N, N) symmetric positive-definite inertia matrix,  N = J * n_f.
    """
    dtype, device = E.dtype, E.device
    J_n = E.shape[0]
    n_f = S.shape[-1]
    N = J_n * n_f

    children = _build_children(parents)
    is_ancestor = _is_ancestor_table(parents)

    # Local Plücker transforms ^i X_parent(i), for i = 0..J-1.
    X_local: list[torch.Tensor] = [torch.eye(6, dtype=dtype, device=device)]
    for i in range(1, J_n):
        X_local.append(plucker_transform(E[i], r[i]))

    # Global transforms ^i X_0 (world → body i), root-first.
    global_X: list[torch.Tensor] = [torch.eye(6, dtype=dtype, device=device)]
    for i in range(1, J_n):
        global_X.append(X_local[i] @ global_X[parents[i]])
    inv_global_X = [inverse_plucker_transform(X) for X in global_X]

    # ── Step 1: Composite rigid-body inertias B_i (bottom-up) ───────────────
    # B_i = sum_{m in subtree(i)} ^i X_m^T I_m ^i X_m  (referred to body i frame)
    # B_i = I_i + sum_{c in children(i)} X^{i→c}^T B_c X^{i→c}
    # where X^{i→c} = ^i X_c = (^c X_i)^{-1} = inverse_plucker_transform(X_local[c]).
    B = I_spatial.clone()  # (J, 6, 6)
    for i in range(J_n - 1, -1, -1):  # leaves → root
        Bi = I_spatial[i]  # start with own inertia
        for c in children[i]:
            X_i_to_c = inverse_plucker_transform(X_local[c])  # ^i X_c
            Bi = Bi + X_i_to_c.T @ B[c] @ X_i_to_c
        B[i] = Bi

    # ── Step 2: Assemble M from B and the ancestor structure ────────────────
    # M[j,k] = S_j^T ^k X_j^T B_k S_k  when j is an ancestor of k.
    # Use symmetry M[k,j] = M[j,k].T for the reverse case.
    M = torch.zeros(N, N, dtype=dtype, device=device)
    S_b = S  # (J, 6, n_f)

    for j in range(J_n):
        Sj = S_b[j]  # (6, n_f)
        for k in range(J_n):
            if not is_ancestor[j][k]:
                continue  # k not in j's subtree → covered by symmetry or zero
            Sk = S_b[k]  # (6, n_f)
            # ^k X_j = global_X[k] @ inv_global_X[j]
            X_k_to_j = global_X[k] @ inv_global_X[j]  # ^k X_j
            block = Sj.T @ X_k_to_j.T @ B[k] @ Sk  # (n_f, n_f)
            j_slice = slice(j * n_f, (j + 1) * n_f)
            k_slice = slice(k * n_f, (k + 1) * n_f)
            M[j_slice, k_slice] = block
            if j != k:
                M[k_slice, j_slice] = block.T  # symmetry

    return M
