"""Mathematical correctness tests for core.homogeneous."""

import pytest
import torch

from hmodel.core.homogeneous import (
    decode_transformation_matrix,
    get_transformation_matrix,
    inverse_transformation_matrix,
    skewv,
    vee,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_rotation(shape, dtype, device):
    """Random rotation matrices via QR decomposition."""
    A = torch.randn(*shape, 3, 3, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(A)
    # Ensure det = +1 (proper rotation)
    sign = torch.linalg.det(Q).sign().unsqueeze(-1).unsqueeze(-1)
    return Q * sign


# ---------------------------------------------------------------------------
# get_transformation_matrix
# ---------------------------------------------------------------------------


class TestGetTransformationMatrix:
    def test_output_shape(self, device, dtype, batch_size):
        R = random_rotation((batch_size,), dtype, device)
        p = torch.randn(batch_size, 3, dtype=dtype, device=device)
        T = get_transformation_matrix(R, p)
        assert T.shape == (batch_size, 4, 4)

    def test_unbatched_shape(self, device, dtype):
        R = random_rotation((), dtype, device)
        p = torch.randn(3, dtype=dtype, device=device)
        T = get_transformation_matrix(R, p)
        assert T.shape == (4, 4)

    def test_rotation_block(self, device, dtype, batch_size):
        R = random_rotation((batch_size,), dtype, device)
        p = torch.randn(batch_size, 3, dtype=dtype, device=device)
        T = get_transformation_matrix(R, p)
        torch.testing.assert_close(T[..., :3, :3], R)

    def test_translation_column(self, device, dtype, batch_size):
        R = random_rotation((batch_size,), dtype, device)
        p = torch.randn(batch_size, 3, dtype=dtype, device=device)
        T = get_transformation_matrix(R, p)
        torch.testing.assert_close(T[..., :3, 3], p)

    def test_bottom_row(self, device, dtype, batch_size):
        R = random_rotation((batch_size,), dtype, device)
        p = torch.randn(batch_size, 3, dtype=dtype, device=device)
        T = get_transformation_matrix(R, p)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dtype, device=device)
        torch.testing.assert_close(T[..., 3, :], expected.expand(batch_size, 4))

    def test_dtype_propagation(self, device):
        R = random_rotation((), torch.float64, device)
        p = torch.randn(3, dtype=torch.float64, device=device)
        T = get_transformation_matrix(R, p)
        assert T.dtype == torch.float64

    def test_gradient_flows_through_rotation(self, device):
        R = random_rotation((), torch.float64, device).requires_grad_(True)
        p = torch.randn(3, dtype=torch.float64, device=device)
        T = get_transformation_matrix(R, p)
        T.sum().backward()
        assert R.grad is not None
        assert not torch.all(R.grad == 0)

    def test_gradient_flows_through_position(self, device):
        R = random_rotation((), torch.float64, device)
        p = torch.randn(3, dtype=torch.float64, device=device, requires_grad=True)
        T = get_transformation_matrix(R, p)
        T.sum().backward()
        assert p.grad is not None
        assert not torch.all(p.grad == 0)


# ---------------------------------------------------------------------------
# decode_transformation_matrix
# ---------------------------------------------------------------------------


class TestDecodeTransformationMatrix:
    def test_roundtrip(self, device, dtype, batch_size):
        R = random_rotation((batch_size,), dtype, device)
        p = torch.randn(batch_size, 3, dtype=dtype, device=device)
        T = get_transformation_matrix(R, p)
        R_out, p_out = decode_transformation_matrix(T)
        torch.testing.assert_close(R_out, R)
        torch.testing.assert_close(p_out, p)

    def test_output_shapes(self, device, dtype, batch_size):
        T = (
            torch.eye(4, dtype=dtype, device=device)
            .unsqueeze(0)
            .expand(batch_size, 4, 4)
        )
        R, p = decode_transformation_matrix(T)
        assert R.shape == (batch_size, 3, 3)
        assert p.shape == (batch_size, 3)


# ---------------------------------------------------------------------------
# inverse_transformation_matrix
# ---------------------------------------------------------------------------


class TestInverseTransformationMatrix:
    def test_t_times_inv_is_identity(self, device, dtype, batch_size):
        R = random_rotation((batch_size,), dtype, device)
        p = torch.randn(batch_size, 3, dtype=dtype, device=device)
        T = get_transformation_matrix(R, p)
        T_inv = inverse_transformation_matrix(T)
        product = torch.matmul(T, T_inv)
        eye = (
            torch.eye(4, dtype=dtype, device=device)
            .unsqueeze(0)
            .expand(batch_size, 4, 4)
        )
        torch.testing.assert_close(product, eye, atol=1e-5, rtol=1e-5)

    def test_inv_times_t_is_identity(self, device, dtype, batch_size):
        R = random_rotation((batch_size,), dtype, device)
        p = torch.randn(batch_size, 3, dtype=dtype, device=device)
        T = get_transformation_matrix(R, p)
        T_inv = inverse_transformation_matrix(T)
        product = torch.matmul(T_inv, T)
        eye = (
            torch.eye(4, dtype=dtype, device=device)
            .unsqueeze(0)
            .expand(batch_size, 4, 4)
        )
        torch.testing.assert_close(product, eye, atol=1e-5, rtol=1e-5)

    def test_pure_rotation_inverse_is_transpose(self, device, dtype):
        R = random_rotation((), dtype, device)
        p = torch.zeros(3, dtype=dtype, device=device)
        T = get_transformation_matrix(R, p)
        T_inv = inverse_transformation_matrix(T)
        R_inv, _ = decode_transformation_matrix(T_inv)
        torch.testing.assert_close(R_inv, R.transpose(-1, -2), atol=1e-6, rtol=1e-6)

    def test_pure_translation_inverse(self, device, dtype):
        R = torch.eye(3, dtype=dtype, device=device)
        p = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=device)
        T = get_transformation_matrix(R, p)
        T_inv = inverse_transformation_matrix(T)
        _, p_inv = decode_transformation_matrix(T_inv)
        torch.testing.assert_close(p_inv, -p, atol=1e-6, rtol=1e-6)

    def test_double_inverse_is_identity(self, device, dtype):
        R = random_rotation((), dtype, device)
        p = torch.randn(3, dtype=dtype, device=device)
        T = get_transformation_matrix(R, p)
        T_inv_inv = inverse_transformation_matrix(inverse_transformation_matrix(T))
        torch.testing.assert_close(T_inv_inv, T, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# skewv
# ---------------------------------------------------------------------------


class TestSkewv:
    def test_output_shape(self, device, dtype, batch_size):
        v = torch.randn(batch_size, 3, dtype=dtype, device=device)
        S = skewv(v)
        assert S.shape == (batch_size, 3, 3)

    def test_antisymmetry(self, device, dtype, batch_size):
        v = torch.randn(batch_size, 3, dtype=dtype, device=device)
        S = skewv(v)
        torch.testing.assert_close(S, -S.transpose(-1, -2), atol=1e-6, rtol=0)

    def test_diagonal_is_zero(self, device, dtype, batch_size):
        v = torch.randn(batch_size, 3, dtype=dtype, device=device)
        S = skewv(v)
        diag = torch.diagonal(S, dim1=-2, dim2=-1)
        torch.testing.assert_close(diag, torch.zeros_like(diag))

    def test_equals_cross_product(self, device, dtype, batch_size):
        """skewv(a) @ b  must equal  a × b for all a, b."""
        a = torch.randn(batch_size, 3, dtype=dtype, device=device)
        b = torch.randn(batch_size, 3, dtype=dtype, device=device)
        cross = torch.linalg.cross(a, b)
        skew_b = torch.matmul(skewv(a), b.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(skew_b, cross, atol=1e-5, rtol=1e-5)

    def test_null_on_self(self, device, dtype, batch_size):
        """skewv(v) @ v == 0 for any v."""
        v = torch.randn(batch_size, 3, dtype=dtype, device=device)
        result = torch.matmul(skewv(v), v.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-6, rtol=0)

    def test_zero_vector(self, device, dtype):
        v = torch.zeros(3, dtype=dtype, device=device)
        S = skewv(v)
        torch.testing.assert_close(S, torch.zeros(3, 3, dtype=dtype, device=device))

    def test_gradient_flows(self, device):
        v = torch.randn(3, dtype=torch.float64, device=device, requires_grad=True)
        S = skewv(v)
        # S.sum() == 0 by antisymmetry; use Frobenius norm squared instead.
        (S * S).sum().backward()
        assert v.grad is not None
        assert not torch.all(v.grad == 0)


# ---------------------------------------------------------------------------
# vee
# ---------------------------------------------------------------------------


class TestVee:
    def test_vee_of_skewv_is_identity(self, device, dtype, batch_size):
        v = torch.randn(batch_size, 3, dtype=dtype, device=device)
        torch.testing.assert_close(vee(skewv(v)), v)

    def test_skewv_of_vee_is_identity(self, device, dtype, batch_size):
        v = torch.randn(batch_size, 3, dtype=dtype, device=device)
        S = skewv(v)
        torch.testing.assert_close(skewv(vee(S)), S)

    def test_output_shape(self, device, dtype, batch_size):
        mat = skewv(torch.randn(batch_size, 3, dtype=dtype, device=device))
        assert vee(mat).shape == (batch_size, 3)
