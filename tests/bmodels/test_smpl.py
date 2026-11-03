"""Mathematical correctness tests for bmodels.smpl.SMPL."""

import pytest
import torch

from hmodel.bmodels.smpl import SMPL

MODEL_PATH = "./models"

# Known constants from the SMPL NPZ files.
N_VERTICES = 6890
N_JOINTS = 24
N_FACES = 13776


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def identity_pose(batch_size: int, n_joints: int, dtype, device) -> torch.Tensor:
    """(..., J, 3, 3) batch of identity rotation matrices."""
    eye = torch.eye(3, dtype=dtype, device=device)
    return eye.unsqueeze(0).unsqueeze(0).expand(batch_size, n_joints, 3, 3)


# ---------------------------------------------------------------------------
# TestSMPLInit
# ---------------------------------------------------------------------------


class TestSMPLInit:
    def test_male_loads(self):
        SMPL(path=MODEL_PATH, gender="male")

    def test_female_loads(self):
        SMPL(path=MODEL_PATH, gender="female")

    def test_invalid_gender_raises(self):
        with pytest.raises(ValueError, match="gender must be"):
            SMPL(path=MODEL_PATH, gender="unknown")

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            SMPL(path="/nonexistent/path", gender="male")

    def test_v_temp_shape(self, smpl_male):
        assert smpl_male.v_temp.shape == (N_VERTICES, 3)

    def test_j_regressor_shape(self, smpl_male):
        assert smpl_male.J_regressor.shape == (N_JOINTS, N_VERTICES)

    def test_skinning_weights_shape(self, smpl_male):
        assert smpl_male.skinning_weights.shape == (N_VERTICES, N_JOINTS)

    def test_kintree_shape(self, smpl_male):
        assert smpl_male.kintree.shape == (2, N_JOINTS)

    def test_faces_shape(self, smpl_male):
        assert smpl_male.faces.shape == (N_FACES, 3)

    def test_mass_shape(self, smpl_male):
        assert smpl_male.mass.shape == (N_JOINTS,)

    def test_com_shape(self, smpl_male):
        assert smpl_male.com.shape == (N_JOINTS, 3)

    def test_inertia_shape(self, smpl_male):
        assert smpl_male.inertia.shape == (N_JOINTS, 3, 3)

    def test_skinning_weights_sum_to_one(self, smpl_male):
        """Each vertex's skinning weights must sum to 1 (partition of unity)."""
        row_sums = smpl_male.skinning_weights.sum(dim=1)  # (V,)
        torch.testing.assert_close(
            row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=0
        )

    def test_skinning_weights_non_negative(self, smpl_male):
        assert (smpl_male.skinning_weights >= 0).all()


# ---------------------------------------------------------------------------
# TestGetZeroState
# ---------------------------------------------------------------------------


class TestGetZeroState:
    def test_output_shapes(self, smpl_male):
        joints, verts = smpl_male.get_zero_state()
        assert joints.shape == (N_JOINTS, 3)
        assert verts.shape == (N_VERTICES, 3)

    def test_root_joint_at_origin(self, smpl_male):
        joints, _ = smpl_male.get_zero_state()
        torch.testing.assert_close(
            joints[0], torch.zeros(3, dtype=joints.dtype), atol=1e-6, rtol=0
        )

    def test_regressor_consistency(self, smpl_male):
        """zero_joints should equal J_regressor @ v_temp, root-centred."""
        joints, _ = smpl_male.get_zero_state()
        expected = smpl_male.J_regressor @ smpl_male.v_temp
        expected = expected - expected[:1]
        torch.testing.assert_close(joints, expected, atol=1e-6, rtol=1e-6)

    def test_vertex_root_centred(self, smpl_male):
        """Vertices should be relative to root joint position."""
        _, verts = smpl_male.get_zero_state()
        raw_root = smpl_male.zero_joints[:1]
        torch.testing.assert_close(
            verts, smpl_male.v_temp - raw_root, atol=1e-6, rtol=1e-6
        )

    def test_deterministic(self, smpl_male):
        j1, v1 = smpl_male.get_zero_state()
        j2, v2 = smpl_male.get_zero_state()
        torch.testing.assert_close(j1, j2)
        torch.testing.assert_close(v1, v2)

    def test_female_different_from_male(self, smpl_male, smpl_female):
        j_m, v_m = smpl_male.get_zero_state()
        j_f, v_f = smpl_female.get_zero_state()
        assert not torch.allclose(j_m, j_f), "Male and female T-poses should differ"
        assert not torch.allclose(v_m, v_f)


# ---------------------------------------------------------------------------
# TestPhysicalProperties
# ---------------------------------------------------------------------------


class TestPhysicalProperties:
    def test_mass_shape(self, smpl_male):
        assert smpl_male.mass.shape == (N_JOINTS,)

    def test_mass_strictly_positive(self, smpl_male):
        """Every joint segment must have positive mass."""
        assert (
            smpl_male.mass > 0
        ).all(), f"Non-positive mass found: {smpl_male.mass[smpl_male.mass <= 0]}"

    def test_total_mass_plausible(self, smpl_male):
        """Total body mass should fall in a realistic human range [40, 120] kg."""
        total = smpl_male.mass.sum().item()
        assert 40.0 <= total <= 120.0, f"Implausible total mass: {total:.1f} kg"

    def test_com_shape(self, smpl_male):
        assert smpl_male.com.shape == (N_JOINTS, 3)

    def test_inertia_shape(self, smpl_male):
        assert smpl_male.inertia.shape == (N_JOINTS, 3, 3)

    def test_inertia_symmetric(self, smpl_male):
        """Inertia tensors must be symmetric: I == I^T."""
        I = smpl_male.inertia
        torch.testing.assert_close(I, I.transpose(-1, -2), atol=1e-5, rtol=1e-5)

    def test_inertia_positive_semi_definite(self, smpl_male):
        """All eigenvalues of each inertia tensor must be >= 0."""
        eigvals = torch.linalg.eigvalsh(smpl_male.inertia)  # (J, 3) real eigenvalues
        min_eigval = eigvals.min().item()
        assert (
            min_eigval >= -1e-5
        ), f"Negative inertia eigenvalue found: {min_eigval:.6f}"

    def test_female_physics_differ(self, smpl_male, smpl_female):
        assert not torch.allclose(
            smpl_male.mass, smpl_female.mass
        ), "Male and female mass distributions should differ"


# ---------------------------------------------------------------------------
# TestForwardKinematics
# ---------------------------------------------------------------------------


class TestForwardKinematics:

    # --- Shape tests ---

    def test_output_shapes_no_mesh(self, smpl_male):
        B = 2
        pose = identity_pose(B, N_JOINTS, torch.float32, torch.device("cpu"))
        r, p, v = smpl_male.forward_kinematics(pose, return_mesh=False)
        assert r.shape == (B, N_JOINTS, 3, 3)
        assert p.shape == (B, N_JOINTS, 3)
        assert v is None

    def test_output_shapes_with_mesh(self, smpl_male):
        B = 2
        pose = identity_pose(B, N_JOINTS, torch.float32, torch.device("cpu"))
        r, p, v = smpl_male.forward_kinematics(pose, return_mesh=True)
        assert r.shape == (B, N_JOINTS, 3, 3)
        assert p.shape == (B, N_JOINTS, 3)
        assert v is not None
        assert v.shape == (B, N_VERTICES, 3)

    def test_vertex_count_matches_template(self, smpl_male):
        B = 1
        pose = identity_pose(B, N_JOINTS, torch.float32, torch.device("cpu"))
        _, _, v = smpl_male.forward_kinematics(pose, return_mesh=True)
        assert v.shape[-2] == smpl_male.v_temp.shape[0]

    def test_unbatched_pose_raises_or_works(self, smpl_male):
        """Unbatched pose (J, 3, 3) should produce (J, 3, 3) and (J, 3) outputs."""
        zero_joints, _ = smpl_male.get_zero_state()
        pose = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(N_JOINTS, 3, 3)
        r, p, _ = smpl_male.forward_kinematics(pose, return_mesh=False)
        assert r.shape == (N_JOINTS, 3, 3)
        assert p.shape == (N_JOINTS, 3)

    # --- Mathematical correctness ---

    def test_identity_pose_gives_tpose_joints(self, smpl_male):
        """Identity rotations → joint positions equal to T-pose positions."""
        zero_joints, _ = smpl_male.get_zero_state()
        B = 1
        pose = identity_pose(B, N_JOINTS, torch.float32, torch.device("cpu"))
        _, global_position, _ = smpl_male.forward_kinematics(pose, return_mesh=False)
        torch.testing.assert_close(
            global_position[0],
            zero_joints.to(global_position.dtype),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_rotation_matrices_stay_orthogonal(self, smpl_male):
        """Output rotation matrices must satisfy R @ R^T ≈ I."""
        B = 3
        # Random valid pose via QR
        torch.manual_seed(0)
        A = torch.randn(B, N_JOINTS, 3, 3, dtype=torch.float32)
        Q, _ = torch.linalg.qr(A)
        sign = torch.linalg.det(Q).sign()[..., None, None]
        pose = Q * sign

        r, _, _ = smpl_male.forward_kinematics(pose, return_mesh=False)
        eye = torch.eye(3, dtype=torch.float32)
        RRt = torch.matmul(r, r.transpose(-1, -2))
        torch.testing.assert_close(RRt, eye.expand_as(RRt), atol=1e-5, rtol=1e-5)

    def test_determinant_is_one(self, smpl_male):
        """Rotation matrices must have det ≈ +1 (proper rotations)."""
        B = 2
        pose = identity_pose(B, N_JOINTS, torch.float32, torch.device("cpu"))
        r, _, _ = smpl_male.forward_kinematics(pose, return_mesh=False)
        dets = torch.linalg.det(r)
        torch.testing.assert_close(dets, torch.ones_like(dets), atol=1e-5, rtol=1e-5)

    def test_root_position_shifts_all_joints(self, smpl_male):
        """Adding root_position must translate every joint (and vertex) by that offset."""
        B = 2
        pose = identity_pose(B, N_JOINTS, torch.float32, torch.device("cpu"))
        root = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).expand(B, 3)

        _, p_base, v_base = smpl_male.forward_kinematics(pose, return_mesh=True)
        _, p_shift, v_shift = smpl_male.forward_kinematics(
            pose, root_position=root, return_mesh=True
        )

        expected_p = p_base + root.unsqueeze(1)
        expected_v = v_base + root.unsqueeze(1)
        torch.testing.assert_close(p_shift, expected_p, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(v_shift, expected_v, atol=1e-6, rtol=1e-6)

    def test_batch_consistency(self, smpl_male):
        """Single-sample result should match the corresponding slice of a batch."""
        torch.manual_seed(42)
        A = torch.randn(4, N_JOINTS, 3, 3, dtype=torch.float32)
        Q, _ = torch.linalg.qr(A)
        sign = torch.linalg.det(Q).sign()[..., None, None]
        pose = Q * sign

        r_batch, p_batch, v_batch = smpl_male.forward_kinematics(pose, return_mesh=True)
        for i in range(4):
            r_i, p_i, v_i = smpl_male.forward_kinematics(pose[i], return_mesh=True)
            torch.testing.assert_close(r_batch[i], r_i, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(p_batch[i], p_i, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(v_batch[i], v_i, atol=1e-5, rtol=1e-5)

    def test_identity_pose_vertices_near_tpose(self, smpl_male):
        """Identity rotations → vertices should match T-pose (up to root centering)."""
        _, zero_vertex = smpl_male.get_zero_state()
        B = 1
        pose = identity_pose(B, N_JOINTS, torch.float32, torch.device("cpu"))
        _, _, global_vertex = smpl_male.forward_kinematics(pose, return_mesh=True)
        torch.testing.assert_close(
            global_vertex[0],
            zero_vertex.to(global_vertex.dtype),
            atol=1e-4,
            rtol=1e-4,
        )
