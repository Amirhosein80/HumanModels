"""Tests for hmodel.core.jacobian — Featherstone body Jacobian."""

from __future__ import annotations

import math

import pytest
import torch

from hmodel.core.jacobian import (
    ancestor_mask,
    body_jacobian,
    body_jacobian_autograd,
    inverse_plucker_transform,
    motion_subspace,
    plucker_transform,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _rot_x(angle: float) -> torch.Tensor:
    """3×3 rotation about X axis."""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)


def _rot_y(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)


def _rot_z(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)


# Simple 3-joint chain: 0 → 1 → 2 → 3
_CHAIN_PARENTS = [-1, 0, 1, 2]

# Y-branch: 0 → 1 → 2, 0 → 3 (two branches from root)
_BRANCH_PARENTS = [-1, 0, 1, 0]


# ── TestPluckerTransform ──────────────────────────────────────────────────────


class TestPluckerTransform:

    def test_identity_no_rotation_no_translation(self):
        """E=I, r=0 → X = I₆."""
        E = torch.eye(3, dtype=torch.float32)
        r = torch.zeros(3, dtype=torch.float32)
        X = plucker_transform(E, r)
        torch.testing.assert_close(
            X, torch.eye(6, dtype=torch.float32), atol=1e-6, rtol=0.0
        )

    def test_round_trip(self):
        """X @ X⁻¹ == I₆ for arbitrary rotation + translation."""
        torch.manual_seed(0)
        E = _rot_x(0.4) @ _rot_y(0.3)
        r = torch.tensor([0.1, 0.2, -0.3], dtype=torch.float32)
        X = plucker_transform(E, r)
        X_inv = inverse_plucker_transform(X)
        product = X @ X_inv
        torch.testing.assert_close(
            product, torch.eye(6, dtype=torch.float32), atol=1e-6, rtol=0.0
        )

    def test_inverse_round_trip(self):
        """X⁻¹ @ X == I₆."""
        E = _rot_z(1.2)
        r = torch.tensor([0.5, -0.1, 0.0], dtype=torch.float32)
        X = plucker_transform(E, r)
        X_inv = inverse_plucker_transform(X)
        product = X_inv @ X
        torch.testing.assert_close(
            product, torch.eye(6, dtype=torch.float32), atol=1e-6, rtol=0.0
        )

    def test_pure_rotation_no_translation(self):
        """r=0: X = block_diag(E, E)."""
        E = _rot_y(0.7)
        r = torch.zeros(3, dtype=torch.float32)
        X = plucker_transform(E, r)
        expected = torch.zeros(6, 6, dtype=torch.float32)
        expected[:3, :3] = E
        expected[3:, 3:] = E
        torch.testing.assert_close(X, expected, atol=1e-6, rtol=0.0)

    def test_batched_shape(self):
        """Arbitrary leading batch dims are preserved."""
        B = 4
        E = torch.eye(3, dtype=torch.float32).expand(B, 3, 3)
        r = torch.zeros(B, 3, dtype=torch.float32)
        X = plucker_transform(E, r)
        assert X.shape == (B, 6, 6)

    def test_gradient_flows(self):
        """gradcheck: gradients flow through E and r."""
        E = _rot_x(0.3).float().requires_grad_(True)
        r = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, requires_grad=True)

        def fn(E, r):
            return plucker_transform(E, r)

        assert torch.autograd.gradcheck(fn, (E, r), eps=1e-4, atol=1e-3)


# ── TestMotionSubspace ────────────────────────────────────────────────────────


class TestMotionSubspace:

    @pytest.mark.parametrize(
        "jt",
        [
            "revolute_x",
            "revolute_y",
            "revolute_z",
            "prismatic_x",
            "prismatic_y",
            "prismatic_z",
        ],
    )
    def test_single_dof_shape(self, jt):
        S = motion_subspace(jt, dtype=torch.float32, device="cpu")
        assert S.shape == (6, 1)

    def test_ball_shape(self):
        S = motion_subspace("ball", dtype=torch.float32, device="cpu")
        assert S.shape == (6, 3)

    def test_revolute_z_value(self):
        S = motion_subspace("revolute_z", dtype=torch.float32, device="cpu")
        expected = torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float32).unsqueeze(-1)
        torch.testing.assert_close(S, expected, atol=0.0, rtol=0.0)

    def test_revolute_x_value(self):
        S = motion_subspace("revolute_x", dtype=torch.float32, device="cpu")
        expected = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32).unsqueeze(-1)
        torch.testing.assert_close(S, expected, atol=0.0, rtol=0.0)

    def test_prismatic_z_value(self):
        S = motion_subspace("prismatic_z", dtype=torch.float32, device="cpu")
        expected = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32).unsqueeze(-1)
        torch.testing.assert_close(S, expected, atol=0.0, rtol=0.0)

    def test_ball_angular_only(self):
        """Ball joint S: top-3 rows = I₃, bottom-3 rows = 0."""
        S = motion_subspace("ball", dtype=torch.float32, device="cpu")
        torch.testing.assert_close(
            S[:3], torch.eye(3, dtype=torch.float32), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            S[3:], torch.zeros(3, 3, dtype=torch.float32), atol=0.0, rtol=0.0
        )

    def test_invalid_joint_type(self):
        with pytest.raises(ValueError, match="Unknown joint type"):
            motion_subspace("unknown", dtype=torch.float32, device="cpu")

    def test_dtype_propagation(self):
        S32 = motion_subspace("revolute_z", dtype=torch.float32, device="cpu")
        assert S32.dtype == torch.float32


# ── TestAncestorMask ──────────────────────────────────────────────────────────


class TestAncestorMask:

    def test_chain_descendants(self):
        """In a 4-joint chain, joint 3 has ancestors {0,1,2,3}."""
        mask = ancestor_mask(_CHAIN_PARENTS)
        assert mask[3, 0] and mask[3, 1] and mask[3, 2] and mask[3, 3]

    def test_chain_non_ancestors(self):
        """Joint 0 has no non-self ancestors in the chain."""
        mask = ancestor_mask(_CHAIN_PARENTS)
        assert not mask[0, 1] and not mask[0, 2] and not mask[0, 3]

    def test_root_self(self):
        """Root is its own ancestor."""
        mask = ancestor_mask(_CHAIN_PARENTS)
        assert mask[0, 0]

    def test_all_include_self(self):
        """Every joint is its own ancestor."""
        mask = ancestor_mask(_CHAIN_PARENTS)
        assert mask.diagonal().all()

    def test_branching_isolation(self):
        """In Y-branch (0→1→2, 0→3): joints 2 and 3 are NOT ancestors of each other."""
        mask = ancestor_mask(_BRANCH_PARENTS)
        assert not mask[2, 3]  # 3 is not ancestor of 2
        assert not mask[3, 2]  # 2 is not ancestor of 3
        assert mask[2, 0]  # 0 is ancestor of 2
        assert mask[3, 0]  # 0 is ancestor of 3

    def test_shape(self):
        mask = ancestor_mask(_CHAIN_PARENTS)
        assert mask.shape == (4, 4)


# ── TestBodyJacobian ──────────────────────────────────────────────────────────


def _make_chain_inputs(J: int = 4, batch=(), dtype=torch.float32):
    """Build a simple chain kinematic tree with identity rotations and unit offsets."""
    E = torch.eye(3, dtype=dtype).expand(*batch, J, 3, 3).clone()
    r = torch.zeros(*batch, J, 3, dtype=dtype)
    r[..., 1:, 2] = 1.0  # all joints offset by 1 along Z (except root at origin)
    parents = [-1] + list(range(J - 1))
    S = torch.stack([motion_subspace("revolute_z", dtype=dtype) for _ in range(J)])
    return E, r, parents, S


class TestBodyJacobian:

    def test_output_shape_unbatched(self):
        E, r, parents, S = _make_chain_inputs(J=4)
        Jac = body_jacobian(E, r, parents, S)
        assert Jac.shape == (4, 6, 4 * 1)  # J=4, n_f=1

    def test_output_shape_batched(self):
        E, r, parents, S = _make_chain_inputs(J=4, batch=(3,))
        Jac = body_jacobian(E, r, parents, S)
        assert Jac.shape == (3, 4, 6, 4)

    def test_output_shape_batched_2d(self):
        E, r, parents, S = _make_chain_inputs(J=4, batch=(2, 3))
        Jac = body_jacobian(E, r, parents, S)
        assert Jac.shape == (2, 3, 4, 6, 4)

    def test_dtype_float32(self):
        E, r, parents, S = _make_chain_inputs(J=4, dtype=torch.float32)
        Jac = body_jacobian(E, r, parents, S)
        assert Jac.dtype == torch.float32

    def test_root_jacobian_own_column_only(self):
        """Root body (index 0): only its own column (j=0) is non-zero."""
        E, r, parents, S = _make_chain_inputs(J=4)
        Jac = body_jacobian(E, r, parents, S)
        root_J = Jac[0]  # (6, 4)
        assert not torch.all(root_J[:, 0] == 0), "root's own column should be non-zero"
        torch.testing.assert_close(
            root_J[:, 1:], torch.zeros(6, 3, dtype=torch.float32), atol=1e-6, rtol=0.0
        )

    def test_leaf_all_ancestor_columns_nonzero(self):
        """Leaf body (index 3 in a chain): all 4 columns should be non-zero."""
        J = 4
        E = torch.eye(3, dtype=torch.float32).expand(J, 3, 3).clone()
        r = torch.zeros(J, 3, dtype=torch.float32)
        # Give each joint a non-zero translation so rel_X is non-trivial
        r[1:, 0] = 0.5
        parents = [-1, 0, 1, 2]
        S = torch.stack([motion_subspace("revolute_z", dtype=torch.float32)] * J)
        Jac = body_jacobian(E, r, parents, S)
        leaf_J = Jac[3]  # (6, 4)
        for j in range(J):
            assert not torch.all(
                leaf_J[:, j] == 0
            ), f"ancestor column {j} should be non-zero"

    def test_non_ancestor_columns_zero(self):
        """In a Y-branch, cross-branch columns must be zero."""
        parents = [-1, 0, 1, 0]
        J = 4
        E = torch.eye(3, dtype=torch.float32).expand(J, 3, 3).clone()
        r = torch.zeros(J, 3, dtype=torch.float32)
        r[1:, 0] = 0.3
        S = torch.stack([motion_subspace("revolute_z", dtype=torch.float32)] * J)
        Jac = body_jacobian(E, r, parents, S)
        # Body 2 (on branch 0→1→2): column 3 (branch 0→3) must be zero
        torch.testing.assert_close(
            Jac[2, :, 3], torch.zeros(6, dtype=torch.float32), atol=1e-6, rtol=0.0
        )
        # Body 3 (on branch 0→3): column 1 and column 2 must be zero
        torch.testing.assert_close(
            Jac[3, :, 1], torch.zeros(6, dtype=torch.float32), atol=1e-6, rtol=0.0
        )
        torch.testing.assert_close(
            Jac[3, :, 2], torch.zeros(6, dtype=torch.float32), atol=1e-6, rtol=0.0
        )

    def test_velocity_consistency_identity_config(self):
        """At identity config, J_i @ dq gives the correct summed spatial velocity.

        For a chain with all E=I and r_i along Z, and revolute-Z joints,
        joint-i body velocity from ancestors j<=i should be:
        - angular: sum of dq[j] along Z for all j<=i
        - linear: sum of r_i-relative contributions (zero for revolute-Z in body frame)
        """
        J = 4
        E, r, parents, S = _make_chain_inputs(J=J)
        Jac = body_jacobian(E, r, parents, S)

        dq = torch.ones(J, 1, dtype=torch.float32)  # unit joint velocities
        dq_flat = dq.reshape(J)  # (J,)

        # For each body, v_body = Jac[body] @ dq_flat
        for body in range(J):
            v = Jac[body] @ dq_flat  # (6,)
            # Revolute-Z sums: angular Z component = number of ancestors (incl. self)
            n_ancestors = body + 1  # chain: ancestors are 0..body
            torch.testing.assert_close(
                v[2],  # angular Z
                torch.tensor(float(n_ancestors), dtype=torch.float32),
                atol=1e-4,
                rtol=0.0,
            )
            # angular X, Y must be zero (revolute-Z only)
            torch.testing.assert_close(
                v[0], torch.tensor(0.0, dtype=torch.float32), atol=1e-6, rtol=0.0
            )
            torch.testing.assert_close(
                v[1], torch.tensor(0.0, dtype=torch.float32), atol=1e-6, rtol=0.0
            )

    def test_batched_vs_single(self):
        """Batch of B must match B individual single-sample calls."""
        B = 4
        J = 3
        torch.manual_seed(7)
        E = torch.stack([_rot_x(0.1 * i) for i in range(J)])  # (J, 3, 3)
        E_batch = E.unsqueeze(0).expand(B, J, 3, 3).clone()
        r = torch.randn(J, 3, dtype=torch.float32) * 0.2
        r_batch = r.unsqueeze(0).expand(B, J, 3).clone()
        parents = [-1, 0, 1]
        S = torch.stack([motion_subspace("revolute_z", dtype=torch.float32)] * J)

        Jac_batch = body_jacobian(E_batch, r_batch, parents, S)  # (B, J, 6, J)
        for b in range(B):
            Jac_single = body_jacobian(E, r, parents, S)
            torch.testing.assert_close(Jac_batch[b], Jac_single, atol=1e-6, rtol=0.0)

    def test_gradient_flows(self):
        """gradcheck: gradients flow through E and r."""
        J = 3
        parents = [-1, 0, 1]
        S = torch.stack([motion_subspace("revolute_z", dtype=torch.float32)] * J)

        E = (
            torch.eye(3, dtype=torch.float32)
            .expand(J, 3, 3)
            .clone()
            .requires_grad_(True)
        )
        r = (torch.randn(J, 3, dtype=torch.float32) * 0.1).requires_grad_(True)

        def fn(E, r):
            return body_jacobian(E, r, parents, S)

        assert torch.autograd.gradcheck(fn, (E, r), eps=1e-4, atol=1e-3)


# ── TestBodyJacobianVsAutograd ────────────────────────────────────────────────


class TestBodyJacobianVsAutograd:
    """Compare the analytical body_jacobian against body_jacobian_autograd.

    body_jacobian_autograd is the reference: it defines the spatial-velocity
    forward pass explicitly and lets PyTorch differentiate it w.r.t. dq.
    body_jacobian should match it to floating-point precision.
    """

    def _compare(self, E, r, parents, S, atol=1e-4):
        analytical = body_jacobian(E, r, parents, S)
        autograd = body_jacobian_autograd(E, r, parents, S)
        torch.testing.assert_close(analytical, autograd, atol=atol, rtol=0.0)

    def test_chain_identity_config(self):
        """Simple 4-joint chain, all E=I."""
        E, r, parents, S = _make_chain_inputs(J=4)
        self._compare(E, r, parents, S)

    def test_chain_arbitrary_rotations(self):
        """4-joint chain with non-trivial rotations and offsets."""
        torch.manual_seed(42)
        J = 4
        E = torch.stack([_rot_x(0.3 * i) @ _rot_y(0.1 * i) for i in range(J)])
        r = torch.randn(J, 3, dtype=torch.float32) * 0.3
        r[0] = 0.0
        parents = [-1, 0, 1, 2]
        S = torch.stack([motion_subspace("revolute_z", dtype=torch.float32)] * J)
        self._compare(E, r, parents, S)

    def test_branching_tree(self):
        """Y-branch topology: 0→1→2, 0→3."""
        torch.manual_seed(7)
        J = 4
        E = torch.stack([_rot_y(0.2 * i) for i in range(J)])
        r = torch.randn(J, 3, dtype=torch.float32) * 0.2
        r[0] = 0.0
        parents = [-1, 0, 1, 0]
        S = torch.stack([motion_subspace("revolute_x", dtype=torch.float32)] * J)
        self._compare(E, r, parents, S)

    def test_ball_joints(self):
        """3-DOF ball joints — n_f=3 per joint."""
        torch.manual_seed(1)
        J = 3
        E = torch.stack([_rot_x(0.5 * i) @ _rot_z(0.2 * i) for i in range(J)])
        r = torch.randn(J, 3, dtype=torch.float32) * 0.25
        r[0] = 0.0
        parents = [-1, 0, 1]
        S = torch.stack([motion_subspace("ball", dtype=torch.float32)] * J)
        # ball joints → (J, 6, J*3) = (3, 6, 9)
        self._compare(E, r, parents, S)

    def test_mixed_joint_types(self):
        """Mixed revolute-Z and prismatic-Z joints."""
        torch.manual_seed(3)
        J = 3
        parents = [-1, 0, 1]
        E = torch.stack([_rot_z(0.4 * i) for i in range(J)])
        r = torch.randn(J, 3, dtype=torch.float32) * 0.15
        r[0] = 0.0
        S = torch.stack(
            [
                motion_subspace("revolute_z", dtype=torch.float32),
                motion_subspace("prismatic_z", dtype=torch.float32),
                motion_subspace("revolute_y", dtype=torch.float32),
            ]
        )
        self._compare(E, r, parents, S)

    def test_longer_chain(self):
        """Longer 6-joint chain."""
        torch.manual_seed(9)
        J = 6
        E = torch.stack([_rot_x(0.15 * i) @ _rot_y(-0.1 * i) for i in range(J)])
        r = torch.randn(J, 3, dtype=torch.float32) * 0.2
        r[0] = 0.0
        parents = [-1, 0, 1, 2, 3, 4]
        S = torch.stack([motion_subspace("revolute_z", dtype=torch.float32)] * J)
        self._compare(E, r, parents, S)
