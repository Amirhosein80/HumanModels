import torch

from hmodel.core.homogeneous import (
    decode_transformation_matrix,
    get_transformation_matrix,
    inverse_transformation_matrix,
)

__all__ = ["ForwardKinematics", "InverseKinematics"]


class ForwardKinematics:
    """Converts local (parent-relative) rotations/positions to global frame.

    Args:
        parents: 1-D sequence of length J-1. parents[i] is the parent joint
                 index of joint i+1.  Joint 0 is the root and has no parent.
    """

    def __init__(self, parents):
        self.parents = parents

    def forward_kinematic_rotation(self, r_local: torch.Tensor) -> torch.Tensor:
        """Accumulate local rotations into global rotations along the kinematic tree.

        Args:
            r_local: (..., J, 3, 3) local rotation matrices

        Returns:
            (..., J, 3, 3) global rotation matrices
        """
        r_global = torch.zeros_like(r_local)
        r_global[..., 0, :, :] = r_local[..., 0, :, :]
        for idx, parent in enumerate(self.parents):
            r_global[..., idx + 1, :, :] = torch.matmul(
                r_global[..., parent, :, :], r_local[..., idx + 1, :, :]
            )
        return r_global

    def bone_to_joint(self, bone_vector: torch.Tensor) -> torch.Tensor:
        """Convert per-joint bone offsets (parent→child vectors) to global positions.

        Args:
            bone_vector: (..., J, 3)  offset from each joint's parent in global frame

        Returns:
            (..., J, 3) global joint positions
        """
        joints_position = torch.zeros_like(bone_vector)
        joints_position[..., 0, :] = bone_vector[..., 0, :]
        for idx, parent in enumerate(self.parents):
            # FIX: accumulate from bone_vector, not from joints_position[idx+1] (which is 0)
            joints_position[..., idx + 1, :] = (
                joints_position[..., parent, :] + bone_vector[..., idx + 1, :]
            )
        return joints_position

    def forward_kinematic(self, r_local: torch.Tensor, p_local: torch.Tensor):
        """Compute global 4x4 transforms from local bone rotations and offsets.

        Args:
            r_local: (..., J, 3, 3) local rotation matrices
            p_local: (..., J, 3)    local bone offset vectors

        Returns:
            global_rotation: (..., J, 3, 3)
            global_position: (..., J, 3)
        """
        local_transformation = get_transformation_matrix(r_local, p_local)
        global_transformation = torch.zeros_like(local_transformation)
        global_transformation[..., 0, :, :] = local_transformation[..., 0, :, :]
        for idx, parent in enumerate(self.parents):
            global_transformation[..., idx + 1, :, :] = torch.matmul(
                global_transformation[..., parent, :, :],
                local_transformation[..., idx + 1, :, :],
            )
        return decode_transformation_matrix(global_transformation)


class InverseKinematics:
    """Converts global rotations/positions back to local (parent-relative) frame.

    Args:
        parents: same convention as ForwardKinematics.
    """

    def __init__(self, parents):
        self.parents = parents

    def inverse_kinematic_rotation(self, r_global: torch.Tensor) -> torch.Tensor:
        """Convert global rotations to local rotations.

        Args:
            r_global: (..., J, 3, 3) global rotation matrices

        Returns:
            (..., J, 3, 3) local rotation matrices
        """
        r_local = torch.zeros_like(r_global)
        r_local[..., 0, :, :] = r_global[..., 0, :, :]
        for idx, parent in enumerate(self.parents):
            r_local[..., idx + 1, :, :] = torch.matmul(
                torch.transpose(r_global[..., parent, :, :], -1, -2),
                r_global[..., idx + 1, :, :],
            )
        return r_local

    def joint_to_bone(self, joints_position: torch.Tensor) -> torch.Tensor:
        """Convert global joint positions to per-joint bone offset vectors.

        Args:
            joints_position: (..., J, 3) global joint positions

        Returns:
            (..., J, 3) bone offset vectors (child position minus parent position)
        """
        bone_vector = torch.zeros_like(joints_position)
        bone_vector[..., 0, :] = joints_position[..., 0, :]
        for idx, parent in enumerate(self.parents):
            bone_vector[..., idx + 1, :] = (
                joints_position[..., idx + 1, :] - joints_position[..., parent, :]
            )
        return bone_vector

    def inverse_kinematic(self, r_global: torch.Tensor, p_global: torch.Tensor):
        """Compute local 4x4 transforms from global rotations and positions.

        Args:
            r_global: (..., J, 3, 3) global rotation matrices
            p_global: (..., J, 3)    global joint positions

        Returns:
            local_rotation: (..., J, 3, 3)
            local_position: (..., J, 3)
        """
        global_transformation = get_transformation_matrix(r_global, p_global)
        local_transformation = torch.zeros_like(global_transformation)
        local_transformation[..., 0, :, :] = global_transformation[..., 0, :, :]
        for idx, parent in enumerate(self.parents):
            # FIX: write into local_transformation (not global_transformation)
            # so the original global frames are preserved for subsequent joints.
            local_transformation[..., idx + 1, :, :] = torch.matmul(
                inverse_transformation_matrix(global_transformation[..., parent, :, :]),
                global_transformation[..., idx + 1, :, :],
            )
        return decode_transformation_matrix(local_transformation)
