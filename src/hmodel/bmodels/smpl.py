import gc
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.interpolate import LinearNDInterpolator

from ..core.homogeneous import get_transformation_matrix, skewv
from ..core.logger import get_logger
from ..kinematics.kinematics import ForwardKinematics, InverseKinematics

logger = get_logger(__name__)


class SMPL:
    """SMPL body model: forward kinematics, LBS skinning, and physical properties.

    Args:
        path:       Directory containing ``SMPL_MALE.npz`` / ``SMPL_FEMALE.npz``.
        gender:     ``"male"`` or ``"female"``.
        calibration: Calibration pose name (currently only ``"T-pose"``).
        resolution: Voxel grid resolution (metres) used by :meth:`physical_properties`.
    """

    def __init__(
        self,
        path: str,
        gender: str = "male",
        calibration: str = "T-pose",
        resolution: float = 0.02,
    ) -> None:
        if gender not in ["male", "female"]:
            logger.error("Invalid gender '%s': must be 'male' or 'female'", gender)
            raise ValueError(f"gender must be 'male' or 'female', got '{gender}'")

        npz_file = os.path.join(path, f"SMPL_{gender.upper()}.npz")
        if not os.path.exists(npz_file):
            logger.error("SMPL model file not found: %s", npz_file)
            raise FileNotFoundError(f"SMPL model file not found: {npz_file}")

        logger.info("SMPL model file loaded: %s", npz_file)

        smpl_params = np.load(npz_file, allow_pickle=True)

        self.v_temp = torch.from_numpy(smpl_params["v_template"])
        self.shapedirs = torch.from_numpy(smpl_params["shapedirs"])
        self.J_regressor = torch.from_numpy(smpl_params["J_regressor"])
        self.kintree = torch.from_numpy(smpl_params["kintree_table"])
        self.skinning_weights = torch.from_numpy(smpl_params["weights"])
        self.faces = torch.from_numpy(smpl_params["f"])
        self.zero_joints = self.J_regressor @ self.v_temp  # (J, 3)
        del smpl_params  # NpzFile no longer needed; release ~50 MB of numpy arrays

        self.parents = self.kintree[0, 1:]

        self.fk = ForwardKinematics(parents=self.parents)
        self.ik = InverseKinematics(parents=self.parents)

        # Cache T-pose state — avoids recomputing two tensor subtractions on every
        # forward_kinematics call.
        self._zero_joints_cached, self._zero_vertex_cached = self._compute_zero_state()

        self.mass, self.com, self.inertia = self.physical_properties(res=resolution)
        gc.collect()  # release trimesh/scipy objects created during physical_properties
        torch.cuda.empty_cache()  # clear fragmented GPU allocator state after heavyweight init

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_zero_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        zero_joints_position = self.zero_joints - self.zero_joints[:1]  # (J, 3)
        zero_vertex_position = self.v_temp - self.zero_joints[:1]  # (V, 3)
        return zero_joints_position, zero_vertex_position

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_zero_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return T-pose joint and vertex positions relative to the root joint.

        Returns:
            zero_joints: (J, 3) joint positions centred at root
            zero_vertex: (V, 3) vertex positions centred at root
        """
        return self._zero_joints_cached, self._zero_vertex_cached

    def physical_properties(self, res: float = 0.02):
        """Compute per-joint mass, centre-of-mass, and inertia tensor.

        Uses a voxel grid at the T-pose to sample interior points and
        distributes mass via skinning weights.

        Args:
            res: Voxel size in metres.

        Returns:
            mass:    (J,)      per-joint mass [kg]
            com:     (J, 3)    per-joint centre-of-mass offset
            inertia: (J, 3, 3) per-joint inertia tensor
        """
        joint, vert = self.get_zero_state()

        mesh = trimesh.Trimesh(
            vertices=vert.numpy(), faces=self.faces.numpy(), process=False
        )

        # Single 3-D grid + one mesh.contains call — no per-slice Python loop.
        mins = vert.min(dim=0).values
        maxs = vert.max(dim=0).values
        gx, gy, gz = torch.meshgrid(
            torch.arange(mins[0].item(), maxs[0].item(), res, dtype=vert.dtype),
            torch.arange(mins[1].item(), maxs[1].item(), res, dtype=vert.dtype),
            torch.arange(mins[2].item(), maxs[2].item(), res, dtype=vert.dtype),
            indexing="ij",
        )
        grid = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # (N, 3)
        mask = torch.from_numpy(mesh.contains(grid.numpy()))
        point = grid[mask]  # (P, 3)
        del grid, mask  # no longer needed after point is extracted

        weight_fn = LinearNDInterpolator(vert.numpy(), self.skinning_weights.numpy())
        weight = torch.from_numpy(weight_fn(point.numpy())).to(vert.dtype).T  # (J, P)
        del weight_fn  # scipy interpolator holds references to large numpy arrays

        weighted_mass = weight * 1000 * res**3  # (J, P)
        mass = weighted_mass.sum(dim=1)  # (J,)

        diff = point.unsqueeze(0) - joint.unsqueeze(1)  # (J, P, 3)
        com = torch.einsum("jp,jpd->jd", weighted_mass, diff) / mass.unsqueeze(
            1
        )  # (J, 3)
        del diff  # (J, P, 3) no longer needed

        local_point_cross = skewv(point.unsqueeze(0) - com.unsqueeze(1))  # (J, P, 3, 3)
        inertia = torch.einsum(
            "jp,jpkl->jkl",
            weighted_mass,
            -(local_point_cross @ local_point_cross),
        )  # (J, 3, 3)
        del local_point_cross  # (J, P, 3, 3) ~5.5 MB, no longer needed

        return mass, com, inertia

    def forward_kinematics(
        self,
        pose: torch.Tensor,
        root_position: Optional[torch.Tensor] = None,
        return_mesh: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Apply forward kinematics and optional LBS mesh deformation.

        Args:
            pose:          (..., J, 3, 3) local joint rotation matrices
            root_position: Optional (..., 3) global root translation
            return_mesh:   If True, also compute and return deformed vertices

        Returns:
            global_rotation: (..., J, 3, 3)
            global_position: (..., J, 3)
            global_vertex:   (..., V, 3) or None
        """
        dtype, device = pose.dtype, pose.device
        zero_joints, zero_vertex = self.get_zero_state()
        zero_joints = zero_joints.to(dtype=dtype, device=device)
        zero_vertex = zero_vertex.to(dtype=dtype, device=device)
        zero_bone_vectors = self.ik.joint_to_bone(zero_joints)  # (J, 3)

        # Broadcast bone vectors to match the batch dims of pose (..., J, 3, 3).
        zero_bone_vectors = zero_bone_vectors.expand_as(pose[..., 0])

        global_rotation, global_position = self.fk.forward_kinematic(
            r_local=pose,
            p_local=zero_bone_vectors,
        )

        if return_mesh:
            global_transformation = get_transformation_matrix(
                global_rotation, global_position
            )
            # (..., J, 4, 4)

            # Compute relative transforms G_j = T_j(pose) @ T_j(rest)^{-1}.
            # T_j(rest) = [I | zero_joints[j]; 0 | 1], so its inverse is [I | -zero_joints[j]; 0 | 1].
            # This keeps the operation autograd-friendly (no in-place mutation).
            n_joints = zero_joints.shape[0]
            eye3 = torch.eye(3, dtype=dtype, device=device)
            rest_T_inv = get_transformation_matrix(
                eye3.unsqueeze(0).expand(n_joints, 3, 3),
                -zero_joints,
            )  # (J, 4, 4)
            G = torch.matmul(global_transformation, rest_T_inv)  # (..., J, 4, 4)
            del global_transformation, rest_T_inv

            # LBS: blend joint relative transforms by per-vertex skinning weights.
            # einsum handles arbitrary batch dims; avoids tensordot + transpose.
            skinning_weights = self.skinning_weights.to(dtype=G.dtype, device=G.device)
            vertex_transformation = torch.einsum(
                "...jkl,vj->...vkl",
                G,  # (..., J, 4, 4)
                skinning_weights,  # (V, J)
            )  # (..., V, 4, 4)

            # Apply to homogeneous vertex coordinates (w=1 for points).
            # F.pad avoids zero-init + in-place fill, keeping gradient path clean.
            vertex_vec = F.pad(zero_vertex, (0, 1), value=1.0)  # (V, 4)
            global_vertex = torch.matmul(
                vertex_transformation,
                vertex_vec[..., None],
            )[
                ..., :3, 0
            ]  # (..., V, 3)
            del vertex_transformation  # (..., V, 4, 4) — up to 1.7 GB @ B=16, free immediately

        else:
            global_vertex = None

        if root_position is not None:
            global_position = global_position + root_position[..., None, :]
            if global_vertex is not None:
                global_vertex = global_vertex + root_position[..., None, :]

        return global_rotation, global_position, global_vertex
