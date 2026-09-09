"""Mathematical correctness tests for core.kinematics."""

import pytest
import torch

from hmodel.kinematics.kinematics import ForwardKinematics, InverseKinematics

# ---------------------------------------------------------------------------
# Helpers & shared tree definitions
# ---------------------------------------------------------------------------


def random_rotation(shape, dtype, device):
    A = torch.randn(*shape, 3, 3, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(A)
    sign = torch.linalg.det(Q).sign().unsqueeze(-1).unsqueeze(-1)
    return Q * sign


# 4-joint chain:  0 → 1 → 2 → 3   (parents of joints 1,2,3 are 0,1,2)
CHAIN_PARENTS = [0, 1, 2]
N_JOINTS = 4  # root + 3 children


# Simple Y-shaped tree:
#      0
#     / \
#    1   2
#    |
#    3
TREE_PARENTS = [0, 0, 1]
N_JOINTS_TREE = 4


@pytest.fixture(params=["chain", "tree"])
def skeleton(request):
    if request.param == "chain":
        return CHAIN_PARENTS, N_JOINTS
    return TREE_PARENTS, N_JOINTS_TREE


# ---------------------------------------------------------------------------
# ForwardKinematics.forward_kinematic_rotation
# ---------------------------------------------------------------------------


class TestFKRotation:
    def test_root_equals_local(self, device, dtype, batch_size):
        fk = ForwardKinematics(CHAIN_PARENTS)
        r_local = random_rotation((batch_size, N_JOINTS), dtype, device)
        r_global = fk.forward_kinematic_rotation(r_local)
        torch.testing.assert_close(r_global[:, 0], r_local[:, 0])

    def test_child_equals_parent_global_times_local(self, device, dtype, batch_size):
        """r_global[j] == r_global[parent(j)] @ r_local[j] for every j."""
        fk = ForwardKinematics(CHAIN_PARENTS)
        r_local = random_rotation((batch_size, N_JOINTS), dtype, device)
        r_global = fk.forward_kinematic_rotation(r_local)
        for idx, parent in enumerate(CHAIN_PARENTS):
            expected = torch.matmul(r_global[:, parent], r_local[:, idx + 1])
            torch.testing.assert_close(
                r_global[:, idx + 1], expected, atol=1e-5, rtol=1e-5
            )

    def test_identity_rotations_give_identity_global(self, device, dtype, batch_size):
        fk = ForwardKinematics(CHAIN_PARENTS)
        eye = torch.eye(3, dtype=dtype, device=device)
        r_local = eye.unsqueeze(0).unsqueeze(0).expand(batch_size, N_JOINTS, 3, 3)
        r_global = fk.forward_kinematic_rotation(r_local)
        expected = eye.unsqueeze(0).unsqueeze(0).expand(batch_size, N_JOINTS, 3, 3)
        torch.testing.assert_close(r_global, expected)

    def test_output_shape(self, device, dtype, batch_size):
        fk = ForwardKinematics(CHAIN_PARENTS)
        r_local = random_rotation((batch_size, N_JOINTS), dtype, device)
        r_global = fk.forward_kinematic_rotation(r_local)
        assert r_global.shape == (batch_size, N_JOINTS, 3, 3)

    def test_tree_topology(self, device, dtype, batch_size):
        """Ensure parent indices other than simple chain are handled correctly."""
        fk = ForwardKinematics(TREE_PARENTS)
        r_local = random_rotation((batch_size, N_JOINTS_TREE), dtype, device)
        r_global = fk.forward_kinematic_rotation(r_local)
        for idx, parent in enumerate(TREE_PARENTS):
            expected = torch.matmul(r_global[:, parent], r_local[:, idx + 1])
            torch.testing.assert_close(
                r_global[:, idx + 1], expected, atol=1e-5, rtol=1e-5
            )


# ---------------------------------------------------------------------------
# InverseKinematics.inverse_kinematic_rotation
# ---------------------------------------------------------------------------


class TestIKRotation:
    def test_roundtrip_fk_ik(self, device, dtype, batch_size):
        """IK(FK(r_local)) ≈ r_local."""
        fk = ForwardKinematics(CHAIN_PARENTS)
        ik = InverseKinematics(CHAIN_PARENTS)
        r_local = random_rotation((batch_size, N_JOINTS), dtype, device)
        r_global = fk.forward_kinematic_rotation(r_local)
        r_local_rec = ik.inverse_kinematic_rotation(r_global)
        torch.testing.assert_close(r_local_rec, r_local, atol=1e-5, rtol=1e-5)

    def test_roundtrip_ik_fk(self, device, dtype, batch_size):
        """FK(IK(r_global)) ≈ r_global."""
        fk = ForwardKinematics(CHAIN_PARENTS)
        ik = InverseKinematics(CHAIN_PARENTS)
        r_global = random_rotation((batch_size, N_JOINTS), dtype, device)
        r_local = ik.inverse_kinematic_rotation(r_global)
        r_global_rec = fk.forward_kinematic_rotation(r_local)
        torch.testing.assert_close(r_global_rec, r_global, atol=1e-5, rtol=1e-5)

    def test_root_unchanged(self, device, dtype, batch_size):
        ik = InverseKinematics(CHAIN_PARENTS)
        r_global = random_rotation((batch_size, N_JOINTS), dtype, device)
        r_local = ik.inverse_kinematic_rotation(r_global)
        torch.testing.assert_close(r_local[:, 0], r_global[:, 0])

    def test_output_shape(self, device, dtype, batch_size):
        ik = InverseKinematics(CHAIN_PARENTS)
        r_global = random_rotation((batch_size, N_JOINTS), dtype, device)
        assert ik.inverse_kinematic_rotation(r_global).shape == (
            batch_size,
            N_JOINTS,
            3,
            3,
        )


# ---------------------------------------------------------------------------
# ForwardKinematics.bone_to_joint / InverseKinematics.joint_to_bone
# ---------------------------------------------------------------------------


class TestBoneJointConversion:
    def test_joint_to_bone_root_unchanged(self, device, dtype, batch_size):
        ik = InverseKinematics(CHAIN_PARENTS)
        joints = torch.randn(batch_size, N_JOINTS, 3, dtype=dtype, device=device)
        bones = ik.joint_to_bone(joints)
        torch.testing.assert_close(bones[:, 0], joints[:, 0])

    def test_bone_is_child_minus_parent(self, device, dtype, batch_size):
        """bone[j] == joint[j] - joint[parent(j)]."""
        ik = InverseKinematics(CHAIN_PARENTS)
        joints = torch.randn(batch_size, N_JOINTS, 3, dtype=dtype, device=device)
        bones = ik.joint_to_bone(joints)
        for idx, parent in enumerate(CHAIN_PARENTS):
            expected = joints[:, idx + 1] - joints[:, parent]
            torch.testing.assert_close(bones[:, idx + 1], expected)

    def test_bone_to_joint_roundtrip(self, device, dtype, batch_size):
        """joint_to_bone(bone_to_joint(bones)) ≈ bones."""
        fk = ForwardKinematics(CHAIN_PARENTS)
        ik = InverseKinematics(CHAIN_PARENTS)
        bones = torch.randn(batch_size, N_JOINTS, 3, dtype=dtype, device=device)
        joints = fk.bone_to_joint(bones)
        bones_rec = ik.joint_to_bone(joints)
        torch.testing.assert_close(bones_rec, bones, atol=1e-5, rtol=1e-5)

    def test_bone_to_joint_roundtrip_inv(self, device, dtype, batch_size):
        """bone_to_joint(joint_to_bone(joints)) ≈ joints."""
        fk = ForwardKinematics(CHAIN_PARENTS)
        ik = InverseKinematics(CHAIN_PARENTS)
        joints = torch.randn(batch_size, N_JOINTS, 3, dtype=dtype, device=device)
        bones = ik.joint_to_bone(joints)
        joints_rec = fk.bone_to_joint(bones)
        torch.testing.assert_close(joints_rec, joints, atol=1e-5, rtol=1e-5)

    def test_bone_to_joint_accumulation(self, device, dtype):
        """For a 3-joint chain with root at origin, joint[k] == sum of bones[0..k]."""
        fk = ForwardKinematics([0, 1])  # 3 joints: 0→1→2
        # Root at origin, two unit offsets along X
        bones = torch.zeros(3, 3, dtype=dtype, device=device)
        bones[0] = torch.tensor([0.0, 0.0, 0.0])
        bones[1] = torch.tensor([1.0, 0.0, 0.0])
        bones[2] = torch.tensor([1.0, 0.0, 0.0])
        joints = fk.bone_to_joint(bones)
        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
        torch.testing.assert_close(joints, expected)

    def test_output_shape(self, device, dtype, batch_size):
        fk = ForwardKinematics(CHAIN_PARENTS)
        ik = InverseKinematics(CHAIN_PARENTS)
        bones = torch.randn(batch_size, N_JOINTS, 3, dtype=dtype, device=device)
        joints = fk.bone_to_joint(bones)
        assert joints.shape == (batch_size, N_JOINTS, 3)
        assert ik.joint_to_bone(joints).shape == (batch_size, N_JOINTS, 3)


# ---------------------------------------------------------------------------
# ForwardKinematics.forward_kinematic / InverseKinematics.inverse_kinematic
# ---------------------------------------------------------------------------


class TestFullKinematicRoundtrip:
    def test_fk_ik_roundtrip(self, device, dtype, batch_size):
        """IK(FK(r_local, p_local)) ≈ (r_local, p_local)."""
        fk = ForwardKinematics(CHAIN_PARENTS)
        ik = InverseKinematics(CHAIN_PARENTS)
        r_local = random_rotation((batch_size, N_JOINTS), dtype, device)
        p_local = torch.randn(batch_size, N_JOINTS, 3, dtype=dtype, device=device)
        r_global, p_global = fk.forward_kinematic(r_local, p_local)
        r_local_rec, p_local_rec = ik.inverse_kinematic(r_global, p_global)
        torch.testing.assert_close(r_local_rec, r_local, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(p_local_rec, p_local, atol=1e-4, rtol=1e-4)

    def test_ik_fk_roundtrip(self, device, dtype, batch_size):
        """FK(IK(r_global, p_global)) ≈ (r_global, p_global)."""
        fk = ForwardKinematics(CHAIN_PARENTS)
        ik = InverseKinematics(CHAIN_PARENTS)
        r_global = random_rotation((batch_size, N_JOINTS), dtype, device)
        p_global = torch.randn(batch_size, N_JOINTS, 3, dtype=dtype, device=device)
        r_local, p_local = ik.inverse_kinematic(r_global, p_global)
        r_global_rec, p_global_rec = fk.forward_kinematic(r_local, p_local)
        torch.testing.assert_close(r_global_rec, r_global, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(p_global_rec, p_global, atol=1e-4, rtol=1e-4)

    def test_identity_pose_output_shape(self, device, dtype, batch_size):
        fk = ForwardKinematics(CHAIN_PARENTS)
        eye = torch.eye(3, dtype=dtype, device=device)
        r_local = eye.expand(batch_size, N_JOINTS, 3, 3)
        p_local = torch.zeros(batch_size, N_JOINTS, 3, dtype=dtype, device=device)
        r_global, p_global = fk.forward_kinematic(r_local, p_local)
        assert r_global.shape == (batch_size, N_JOINTS, 3, 3)
        assert p_global.shape == (batch_size, N_JOINTS, 3)

    def test_identity_pose_gives_identity_rotations(self, device, dtype, batch_size):
        """All-identity local rotations → all-identity global rotations."""
        fk = ForwardKinematics(CHAIN_PARENTS)
        eye = torch.eye(3, dtype=dtype, device=device)
        r_local = eye.expand(batch_size, N_JOINTS, 3, 3)
        p_local = torch.zeros(batch_size, N_JOINTS, 3, dtype=dtype, device=device)
        r_global, _ = fk.forward_kinematic(r_local, p_local)
        torch.testing.assert_close(r_global, eye.expand(batch_size, N_JOINTS, 3, 3))
