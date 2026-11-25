import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import trimesh
from scipy.interpolate import LinearNDInterpolator

from ..core.homogeneous import skewv
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

        self.gender = gender
        self._model_path = path

        npz_file = os.path.join(path, f"SMPL_{gender.upper()}.npz")
        self._npz_path = npz_file
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

        # Physical properties with persistent disk cache (avoids the ~10 s
        # trimesh/scipy voxel computation on every init).
        cached = self._load_cached_physical_properties(resolution)
        if cached is not None:
            self.mass, self.com, self.inertia = cached
            logger.info("Loaded physical properties from cache")
        else:
            self.mass, self.com, self.inertia = self.physical_properties(res=resolution)
            self._save_cached_physical_properties(resolution)
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
    # Physical-properties persistent cache
    # ------------------------------------------------------------------

    _CACHE_SCHEMA = 1

    def _cache_dir(self) -> Path:
        """Directory for cached physical properties.

        Stored next to the model files so caches travel with the assets, but
        falls back to the user cache dir if the models directory is read-only.
        """
        model_dir = (
            Path(self._npz_path).parent if hasattr(self, "_npz_path") else Path(".")
        )
        candidates = [model_dir / ".hmodel_cache", Path.home() / ".cache" / "hmodel"]
        for d in candidates:
            try:
                d.mkdir(parents=True, exist_ok=True)
                # Verify writable
                probe = d / ".write_test"
                probe.touch()
                probe.unlink()
                return d
            except OSError:
                continue
        # Last resort: current working directory
        fallback = Path.cwd() / ".hmodel_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def _cache_key(self, resolution: float) -> str:
        """Stable cache key incorporating gender, asset identity, and config.

        Invalidates when the npz file changes (mtime/size), the gender changes,
        the resolution changes, or the computation schema changes.
        """
        npz_file = os.path.join(self._model_path, f"SMPL_{self.gender.upper()}.npz")
        st = os.stat(npz_file)
        h = hashlib.sha256()
        h.update(str(self._CACHE_SCHEMA).encode())
        h.update(self.gender.encode())
        h.update(str(resolution).encode())
        h.update(str(st.st_mtime_ns).encode())
        h.update(str(st.st_size).encode())
        return h.hexdigest()[:16]

    def _cache_path(self, resolution: float) -> Path:
        return (
            self._cache_dir()
            / f"physprops_{self.gender}_{self._cache_key(resolution)}.npz"
        )

    def _load_cached_physical_properties(
        self, resolution: float
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Load cached mass/com/inertia if a valid cache file exists."""
        path = self._cache_path(resolution)
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                if int(data["schema"]) != self._CACHE_SCHEMA:
                    return None
                mass = torch.from_numpy(data["mass"])
                com = torch.from_numpy(data["com"])
                inertia = torch.from_numpy(data["inertia"])
            return mass, com, inertia
        except Exception:
            # Corrupt cache — ignore and recompute
            return None

    def _save_cached_physical_properties(self, resolution: float) -> None:
        """Persist mass/com/inertia to the cache directory."""
        path = self._cache_path(resolution)
        try:
            np.savez_compressed(
                path,
                schema=self._CACHE_SCHEMA,
                mass=self.mass.numpy(),
                com=self.com.numpy(),
                inertia=self.inertia.numpy(),
            )
        except OSError:
            # Cache dir not writable — skip caching, computation is still correct
            pass

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
            # LBS: v' = sum_j w_vj * (R_j @ v + t_j)
            #      = (sum_j w_vj * R_j) @ v + sum_j w_vj * t_j
            #
            # G_j = T_global_j @ T_rest_j^{-1} where T_rest_j^{-1} = [I | -zero_joints[j]; 0|1].
            # So G_j = [R_global_j | p_global_j - R_global_j @ zero_joints[j]; 0|1].
            # We extract rotation + translation directly, avoiding the (..., J, 4, 4)
            # relative-transform build and the (..., V, 4, 4) per-vertex 4x4 blend.
            # G rotation part is just global_rotation; translation part is:
            #   t_G[j] = p_global[j] - R_global[j] @ zero_joints[j]
            rot_part = global_rotation  # (..., J, 3, 3)
            t_part = global_position - torch.matmul(
                global_rotation, zero_joints.unsqueeze(-1)
            ).squeeze(
                -1
            )  # (..., J, 3)

            skinning_weights = self.skinning_weights.to(
                dtype=rot_part.dtype, device=rot_part.device
            )  # (V, J)

            # Blend rotations and translations separately (both 3x3 / 3, not 4x4).
            rot_v = torch.einsum(
                "...jik,vj->...vik",
                rot_part,  # (..., J, 3, 3)  j=joint, i,k=matrix
                skinning_weights,  # (V, J)    v=vertex, j=joint
            )  # (..., V, 3, 3)
            trans_v = torch.einsum(
                "...ji,vj->...vi",
                t_part,  # (..., J, 3)   j=joint, i=coord
                skinning_weights,  # (V, J)    v=vertex, j=joint
            )  # (..., V, 3)

            # Apply blended transform to root-centred rest vertices:
            #   v'[v] = sum_k rot_v[v, i, k] * zero_vertex[v, k] + trans_v[v, i]
            global_vertex = (
                torch.einsum("...vik,vk->...vi", rot_v, zero_vertex) + trans_v
            )  # (..., V, 3)
            del rot_v, trans_v

        else:
            global_vertex = None

        if root_position is not None:
            global_position = global_position + root_position[..., None, :]
            if global_vertex is not None:
                global_vertex = global_vertex + root_position[..., None, :]

        return global_rotation, global_position, global_vertex
