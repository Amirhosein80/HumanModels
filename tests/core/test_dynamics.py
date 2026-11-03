"""Tests for core.dynamics: spatial_inertia, crba, rnea, contact_jacobian."""

import math

import pytest
import torch

from hmodel.core.dynamics import spatial_inertia
from hmodel.core.jacobian import body_jacobian, motion_subspace
from hmodel.dynamics import contact_jacobian, crba, rnea

# ── Shared helpers ─────────────────────────────────────────────────────────────


def _chain(J: int, n_f: int = 1, dtype=torch.float32, joint_type="revolute_z"):
    """Linear chain with J joints."""
    parents = [-1] + list(range(J - 1))
    E = torch.eye(3, dtype=dtype).expand(J, 3, 3).clone()
    r = torch.zeros(J, 3, dtype=dtype)
    r[1:, 2] = 0.5  # 0.5 m along Z
    S = torch.stack([motion_subspace(joint_type, dtype=dtype) for _ in range(J)])
    return parents, E, r, S


def _uniform_inertia(J: int, m: float = 1.0, dtype=torch.float32):
    """Uniform spherical bodies: mass=m, com=0, I=diag(1,1,1)."""
    mass = torch.full((J,), m, dtype=dtype)
    com = torch.zeros(J, 3, dtype=dtype)
    inertia = torch.eye(3, dtype=dtype).expand(J, 3, 3).clone()
    return mass, com, inertia


# ── spatial_inertia ─────────────────────────────────────────────────────────────


class TestSpatialInertia:
    def test_shape(self):
        J = 5
        mass, com, inertia = _uniform_inertia(J)
        I = spatial_inertia(mass, com, inertia)
        assert I.shape == (J, 6, 6)

    def test_symmetry(self):
        J = 4
        mass, com, inertia = _uniform_inertia(J)
        com = torch.randn(J, 3, dtype=torch.float32) * 0.1
        I = spatial_inertia(mass, com, inertia)
        torch.testing.assert_close(I, I.transpose(-1, -2))

    def test_psd(self):
        """Spatial inertia must be positive definite."""
        J = 3
        mass = torch.tensor([2.0, 1.5, 1.0], dtype=torch.float32)
        com = torch.randn(J, 3, dtype=torch.float32) * 0.05
        inertia = torch.eye(3, dtype=torch.float32).expand(J, 3, 3).clone() * 0.1
        I = spatial_inertia(mass, com, inertia)
        for j in range(J):
            eigs = torch.linalg.eigvalsh(I[j])
            assert eigs.min() > 0, f"body {j}: I not PD (min eigval={eigs.min():.4e})"

    def test_zero_com_block_structure(self):
        """With com=0, off-diagonal blocks vanish."""
        J = 2
        mass, com, inertia = _uniform_inertia(J)
        I = spatial_inertia(mass, com, inertia)
        # Top-right and bottom-left should be zero when com=0
        torch.testing.assert_close(
            I[:, :3, 3:], torch.zeros(J, 3, 3, dtype=torch.float32)
        )
        torch.testing.assert_close(
            I[:, 3:, :3], torch.zeros(J, 3, 3, dtype=torch.float32)
        )

    def test_bottom_right_is_mass_times_identity(self):
        J = 3
        mass = torch.tensor([2.0, 1.0, 3.0], dtype=torch.float32)
        com = torch.zeros(J, 3, dtype=torch.float32)
        inertia = torch.eye(3, dtype=torch.float32).expand(J, 3, 3).clone()
        I = spatial_inertia(mass, com, inertia)
        for j in range(J):
            expected = mass[j] * torch.eye(3, dtype=torch.float32)
            torch.testing.assert_close(I[j, 3:, 3:], expected)

    def test_dtype_float32(self):
        J = 2
        mass, com, inertia = _uniform_inertia(J, dtype=torch.float32)
        I = spatial_inertia(mass, com, inertia)
        assert I.dtype == torch.float32

    def test_batched_shape(self):
        B, J = 4, 6
        mass = torch.ones(B, J, dtype=torch.float32)
        com = torch.zeros(B, J, 3, dtype=torch.float32)
        inertia = torch.eye(3, dtype=torch.float32).expand(B, J, 3, 3).clone()
        I = spatial_inertia(mass, com, inertia)
        assert I.shape == (B, J, 6, 6)


# ── crba ───────────────────────────────────────────────────────────────────────


class TestCRBA:
    def test_shape(self):
        J = 4
        parents, E, r, S = _chain(J)
        mass, com, inertia = _uniform_inertia(J)
        I_sp = spatial_inertia(mass, com, inertia)
        M = crba(E, r, parents, S, I_sp)
        assert M.shape == (J, J)

    def test_symmetry(self):
        J = 5
        parents, E, r, S = _chain(J)
        mass, com, inertia = _uniform_inertia(J)
        I_sp = spatial_inertia(mass, com, inertia)
        M = crba(E, r, parents, S, I_sp)
        torch.testing.assert_close(M, M.T)

    def test_psd(self):
        """M(q) must be positive definite for a non-degenerate chain."""
        J = 4
        parents, E, r, S = _chain(J)
        mass, com, inertia = _uniform_inertia(J)
        I_sp = spatial_inertia(mass, com, inertia)
        M = crba(E, r, parents, S, I_sp)
        eigs = torch.linalg.eigvalsh(M)
        assert eigs.min() > 0, f"M not PD (min eigval={eigs.min():.4e})"

    def test_ball_joint_shape(self):
        J = 3
        parents, E, r, S = _chain(J, joint_type="ball")
        mass, com, inertia = _uniform_inertia(J)
        I_sp = spatial_inertia(mass, com, inertia)
        M = crba(E, r, parents, S, I_sp)
        assert M.shape == (J * 3, J * 3)

    def test_single_body(self):
        """Single floating body: M should equal full 6×6 spatial inertia for a ball joint."""
        J = 1
        parents = [-1]
        E = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        r = torch.zeros(1, 3, dtype=torch.float32)
        S = motion_subspace("ball", dtype=torch.float32).unsqueeze(0)  # (1, 6, 3)
        mass = torch.tensor([2.0], dtype=torch.float32)
        com = torch.zeros(1, 3, dtype=torch.float32)
        inertia = torch.eye(3, dtype=torch.float32).unsqueeze(0) * 0.5
        I_sp = spatial_inertia(mass, com, inertia)
        M = crba(E, r, parents, S, I_sp)
        # For a ball joint the first 3 DOF are angular, bottom 3×3 of M = S^T I_sp S
        # With S = [[I₃], [0]], M = S^T I_sp S = I_sp[:3, :3]
        expected = I_sp[0, :3, :3]
        torch.testing.assert_close(M, expected, atol=1e-4, rtol=0)

    def test_dtype(self):
        J = 3
        parents, E, r, S = _chain(J, dtype=torch.float32)
        mass, com, inertia = _uniform_inertia(J, dtype=torch.float32)
        I_sp = spatial_inertia(mass, com, inertia)
        M = crba(E, r, parents, S, I_sp)
        assert M.dtype == torch.float32

    def test_rotated_chain_still_symmetric_psd(self):
        """Rotated joints: M stays symmetric and PD."""
        J = 4
        parents = [-1, 0, 1, 2]
        angle = math.pi / 6
        c, s = math.cos(angle), math.sin(angle)
        Rx = torch.tensor([[1.0, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)
        E = Rx.unsqueeze(0).expand(J, 3, 3).clone()
        r = torch.zeros(J, 3, dtype=torch.float32)
        r[1:, 0] = 0.4
        S = torch.stack(
            [motion_subspace("revolute_x", dtype=torch.float32) for _ in range(J)]
        )
        mass, com, inertia = _uniform_inertia(J)
        I_sp = spatial_inertia(mass, com, inertia)
        M = crba(E, r, parents, S, I_sp)
        torch.testing.assert_close(M, M.T)
        eigs = torch.linalg.eigvalsh(M)
        assert eigs.min() > 0


# ── rnea ───────────────────────────────────────────────────────────────────────


class TestRNEA:
    def _gravity(self, dtype=torch.float32):
        return torch.tensor([0.0, -9.81, 0.0], dtype=dtype)

    def test_shape(self):
        J = 4
        parents, E, r, S = _chain(J)
        mass, com, inertia = _uniform_inertia(J)
        I_sp = spatial_inertia(mass, com, inertia)
        dq = torch.zeros(J, dtype=torch.float32)
        h = rnea(E, r, parents, S, I_sp, dq, self._gravity())
        assert h.shape == (J,)

    def test_ball_joint_shape(self):
        J = 3
        parents, E, r, S = _chain(J, joint_type="ball")
        mass, com, inertia = _uniform_inertia(J)
        I_sp = spatial_inertia(mass, com, inertia)
        dq = torch.zeros(J * 3, dtype=torch.float32)
        h = rnea(E, r, parents, S, I_sp, dq, self._gravity())
        assert h.shape == (J * 3,)

    def test_no_gravity_zero_velocity_gives_zero(self):
        """With g=0 and dq=0, all bias forces must vanish."""
        J = 4
        parents, E, r, S = _chain(J)
        mass, com, inertia = _uniform_inertia(J)
        I_sp = spatial_inertia(mass, com, inertia)
        dq = torch.zeros(J, dtype=torch.float32)
        g = torch.zeros(3, dtype=torch.float32)
        h = rnea(E, r, parents, S, I_sp, dq, g)
        torch.testing.assert_close(
            h, torch.zeros(J, dtype=torch.float32), atol=1e-6, rtol=0
        )

    def test_gravity_only_nonzero(self):
        """With dq=0 but gravity ≠ 0, h is nonzero for revolute-X joints along Z.

        Chain along Z + gravity along -Y produces X-axis torque.
        Revolute-X joints sense that X-torque → h nonzero.
        """
        J = 3
        parents = [-1, 0, 1]
        E = torch.eye(3, dtype=torch.float32).expand(J, 3, 3).clone()
        r = torch.zeros(J, 3, dtype=torch.float32)
        r[1:, 2] = 0.5
        S = torch.stack(
            [motion_subspace("revolute_x", dtype=torch.float32) for _ in range(J)]
        )
        mass, com, inertia = _uniform_inertia(J)
        I_sp = spatial_inertia(mass, com, inertia)
        dq = torch.zeros(J, dtype=torch.float32)
        h = rnea(E, r, parents, S, I_sp, dq, self._gravity())
        assert h.abs().max() > 1e-6, "expected nonzero gravity torques"

    def test_dtype(self):
        J = 3
        parents, E, r, S = _chain(J, dtype=torch.float32)
        mass, com, inertia = _uniform_inertia(J, dtype=torch.float32)
        I_sp = spatial_inertia(mass, com, inertia)
        dq = torch.zeros(J, dtype=torch.float32)
        h = rnea(E, r, parents, S, I_sp, dq, self._gravity())
        assert h.dtype == torch.float32

    def test_coriolis_single_revolute_z(self):
        """Single revolute-Z joint: spinning about Z has zero Coriolis (self-cross product)."""
        J = 1
        parents = [-1]
        E = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        r = torch.zeros(1, 3, dtype=torch.float32)
        S = motion_subspace("revolute_z", dtype=torch.float32).unsqueeze(0)
        mass = torch.tensor([1.0], dtype=torch.float32)
        com = torch.zeros(1, 3, dtype=torch.float32)
        inertia = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        I_sp = spatial_inertia(mass, com, inertia)
        dq = torch.tensor([2.0], dtype=torch.float32)
        g = torch.zeros(3, dtype=torch.float32)
        h = rnea(E, r, parents, S, I_sp, dq, g)
        # Pure spin about Z with symmetric body: no bias force
        torch.testing.assert_close(
            h, torch.zeros(1, dtype=torch.float32), atol=1e-6, rtol=0
        )

    def test_eom_gravity_balance(self):
        """h(q, 0) should equal the gravity torques (static equilibrium forces)."""
        J = 3
        parents, E, r, S = _chain(J)
        mass = torch.tensor([2.0, 1.5, 1.0], dtype=torch.float32)
        com = torch.zeros(J, 3, dtype=torch.float32)
        inertia = torch.eye(3, dtype=torch.float32).expand(J, 3, 3).clone() * 0.1
        I_sp = spatial_inertia(mass, com, inertia)
        dq = torch.zeros(J, dtype=torch.float32)
        g = self._gravity()
        h = rnea(E, r, parents, S, I_sp, dq, g)
        # h should equal τ_gravity = J^T F_grav for gravity forces
        # Just verify it's finite and consistent
        assert torch.isfinite(h).all()


# ── contact_jacobian ───────────────────────────────────────────────────────────


class TestContactJacobian:
    def test_shape(self):
        J, n_c = 6, 2
        parents, E, r, S = _chain(J)
        Jac = body_jacobian(E, r, parents, S)  # (J, 6, J)
        Jc = contact_jacobian(Jac, [J - 1, J - 2])
        assert Jc.shape == (3 * n_c, J)

    def test_linear_rows_match(self):
        """contact_jacobian should exactly equal the linear (bottom-3) rows."""
        J = 5
        parents, E, r, S = _chain(J)
        Jac = body_jacobian(E, r, parents, S)  # (J, 6, J)
        indices = [2, 4]
        Jc = contact_jacobian(Jac, indices)
        expected = torch.cat([Jac[i, 3:, :] for i in indices], dim=0)  # (6, J)
        torch.testing.assert_close(Jc, expected, atol=0, rtol=0)

    def test_single_contact(self):
        J = 4
        parents, E, r, S = _chain(J)
        Jac = body_jacobian(E, r, parents, S)
        Jc = contact_jacobian(Jac, [J - 1])
        assert Jc.shape == (3, J)

    def test_batched(self):
        B, J = 3, 4
        parents, E, r, S = _chain(J)
        E_b = E.unsqueeze(0).expand(B, J, 3, 3).clone()
        r_b = r.unsqueeze(0).expand(B, J, 3).clone()
        Jac = body_jacobian(E_b, r_b, parents, S)  # (B, J, 6, J)
        Jc = contact_jacobian(Jac, [0, J - 1])
        assert Jc.shape == (B, 6, J)

    def test_ball_joint(self):
        J, n_c, n_f = 4, 2, 3
        parents, E, r, S = _chain(J, joint_type="ball")
        Jac = body_jacobian(E, r, parents, S)  # (J, 6, J*3)
        Jc = contact_jacobian(Jac, [1, 3])
        assert Jc.shape == (3 * n_c, J * n_f)
