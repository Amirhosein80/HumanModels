"""rerun-based SMPL body model visualizer with full mesh rendering.

SMPL coordinate system: **X = right, Y = up, Z = forward**.

Online usage (real-time loop)::

    from hmodel.viz import SMPLVisualizer
    import time

    viz = SMPLVisualizer(smpl)
    viz.show_frame(pose, root_position, block=False)   # opens viewer
    while True:
        pose = get_next_pose()          # (24, 3, 3)
        viz.update(pose, root_position)
        time.sleep(1 / 30)

Offline usage::

    viz.animate(poses, root_positions, fps=30)         # scrubable timeline
    viz.save_animation(poses, root_positions, "walk.rrd")
    # replay: rerun walk.rrd
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import rerun as rr
import torch

from ..bmodels.smpl import SMPL

N_JOINTS = 24

# Colours (RGBA 0-1)
_C_BODY = [0.76, 0.75, 0.80, 1.00]
_C_BONE = [0.28, 0.60, 1.00, 1.00]
_C_JOINT = [1.00, 0.30, 0.20, 1.00]
_C_ROOT = [1.00, 0.65, 0.10, 1.00]
_C_FLOOR0 = [0.82, 0.82, 0.82]
_C_FLOOR1 = [0.68, 0.68, 0.68]
_C_TRAJ = [0.50, 0.50, 0.90, 1.00]


def _to_rgba8(c: list) -> list[int]:
    """Convert 0-1 float colour to 0-255 uint8 RGBA."""
    rgb = [int(v * 255) for v in c[:3]]
    a = int(c[3] * 255) if len(c) == 4 else 255
    return rgb + [a]


def _make_checkerboard(
    size: float = 6.0,
    n: int = 12,
    y: float = 0.0,
    cx: float = 0.0,
    cz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """XZ-plane checkerboard centred at (cx, y, cz) (SMPL: Y is up).

    Returns:
        vertices:  (V, 3) float32
        triangles: (F, 3) uint32
        colors:    (V, 4) uint8 RGBA
    """
    cell = size / n
    verts, faces, colors = [], [], []
    vi = 0
    for i in range(n):
        for j in range(n):
            x0 = cx - size / 2 + i * cell
            z0 = cz - size / 2 + j * cell
            x1 = x0 + cell
            z1 = z0 + cell
            verts += [[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]]
            faces += [[vi, vi + 1, vi + 2], [vi, vi + 2, vi + 3]]
            vi += 4
            col = _C_FLOOR0 if (i + j) % 2 == 0 else _C_FLOOR1
            rgba = [int(v * 255) for v in col] + [255]
            colors += [rgba] * 4
    return (
        np.array(verts, dtype=np.float32),
        np.array(faces, dtype=np.uint32),
        np.array(colors, dtype=np.uint8),
    )


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals by accumulating face normals."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    vn = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return (vn / norms).astype(np.float32)


def _validate_sequence(
    poses: torch.Tensor, root_positions: Optional[torch.Tensor]
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if poses.ndim != 4 or poses.shape[-3:] != (N_JOINTS, 3, 3):
        raise ValueError(
            f"poses must be (T, {N_JOINTS}, 3, 3), got {tuple(poses.shape)}"
        )
    T = poses.shape[0]
    if root_positions is not None and root_positions.shape != (T, 3):
        raise ValueError(
            f"root_positions must be ({T}, 3), got {tuple(root_positions.shape)}"
        )
    return poses, root_positions


class SMPLVisualizer:
    """rerun-based SMPL visualizer with full mesh and skeleton rendering.

    SMPL coordinate convention: **X = right, Y = up, Z = forward**.
    The rerun viewer opens automatically (native desktop app) on first use.

    Args:
        smpl:           Loaded :class:`~hmodel.bmodels.smpl.SMPL` instance.
        show_mesh:      Render the full 6890-vertex body mesh.
        show_skeleton:  Render joint points and bone line strips.
        show_floor:     Draw a checkerboard ground plane at foot level.
        recording_id:   rerun application/recording ID shown in the viewer title.
    """

    def __init__(
        self,
        smpl: SMPL,
        show_mesh: bool = True,
        show_skeleton: bool = True,
        show_floor: bool = True,
        recording_id: str = "SMPL Visualizer  |  X=right  Y=up  Z=forward",
    ) -> None:
        self._smpl = smpl
        self._show_mesh = show_mesh
        self._show_skeleton = show_skeleton
        self._faces = smpl.faces.numpy().astype(np.uint32)

        # Kinematic edge list: (parent_idx, child_idx)
        self._edges: list[tuple[int, int]] = [
            (int(smpl.kintree[0, j + 1]), j + 1) for j in range(N_JOINTS - 1)
        ]

        # Foot-level Y from T-pose (SMPL: Y is up)
        zero_joints, _ = smpl.get_zero_state()
        self._floor_y = float(zero_joints[:, 1].min().item()) - 0.02

        # Initialise rerun; spawn=True opens the native desktop viewer
        rr.init(recording_id, spawn=True)

        # Declare world coordinate system: X=right, Y=up, Z=forward
        rr.log(
            "world",
            rr.ViewCoordinates.RIGHT_HAND_Y_UP,
            static=True,
        )

        # Static floor logged once (large default; animate() overrides with trajectory bounds)
        if show_floor:
            verts, tris, colors = _make_checkerboard(size=10.0, n=20, y=self._floor_y)
            rr.log(
                "world/floor",
                rr.Mesh3D(
                    vertex_positions=verts,
                    triangle_indices=tris,
                    vertex_colors=colors,
                ),
                static=True,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """Always True — rerun viewer runs independently; there is no close signal."""
        return True

    def show_frame(
        self,
        pose: torch.Tensor,
        root_position: Optional[torch.Tensor] = None,
        block: bool = True,
    ) -> None:
        """Render a single pose frame in the rerun viewer.

        Args:
            pose:          (24, 3, 3) local joint rotation matrices.
            root_position: Optional (3,) global root translation in metres.
            block:         If True, block until Ctrl-C (gives time to inspect
                           the frame). Set ``False`` for a real-time loop.
        """
        if tuple(pose.shape) != (N_JOINTS, 3, 3):
            raise ValueError(f"pose must be (24, 3, 3), got {tuple(pose.shape)}")
        if root_position is not None and tuple(root_position.shape) != (3,):
            raise ValueError(
                f"root_position must be (3,), got {tuple(root_position.shape)}"
            )
        joints, vertices = self._forward(pose, root_position)
        self._log_frame(joints, vertices, root_position)

        if block:
            print("Frame logged to rerun. Press Ctrl-C to exit.")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass

    def update(
        self,
        pose: torch.Tensor,
        root_position: Optional[torch.Tensor] = None,
    ) -> None:
        """Push a new pose into the live viewer (non-blocking).

        Call this inside a real-time loop after :meth:`show_frame`.

        Args:
            pose:          (24, 3, 3) local joint rotation matrices.
            root_position: Optional (3,) global root translation.
        """
        joints, vertices = self._forward(pose, root_position)
        self._log_frame(joints, vertices, root_position)

    def animate(
        self,
        poses: torch.Tensor,
        root_positions: Optional[torch.Tensor] = None,
        fps: float = 30.0,
        repeat: bool = True,
    ) -> None:
        """Log a pose sequence so the rerun timeline is scrubable.

        The viewer's built-in timeline scrubber lets you pause, seek, and
        replay the sequence interactively. Returns after logging all frames.

        Args:
            poses:          (T, 24, 3, 3) local joint rotation matrices.
            root_positions: Optional (T, 3) root translations per frame.
            fps:            Frames per second for the rerun timeline.
            repeat:         Unused (rerun timeline is always re-playable).
        """
        poses, root_positions = _validate_sequence(poses, root_positions)
        T = poses.shape[0]

        # Log root trajectory and a floor sized to cover the full path
        if root_positions is not None:
            pts = root_positions.cpu().numpy().astype(np.float32)
            pts_floor = pts.copy()
            pts_floor[:, 1] = self._floor_y
            rr.log(
                "world/trajectory",
                rr.LineStrips3D(
                    [pts_floor],
                    colors=[_to_rgba8(_C_TRAJ)],
                    radii=0.006,
                ),
                static=True,
            )
            # Re-log floor centred on trajectory midpoint with enough margin
            pad = 2.0
            cx = float((pts[:, 0].min() + pts[:, 0].max()) / 2)
            cz = float((pts[:, 2].min() + pts[:, 2].max()) / 2)
            span_x = float(pts[:, 0].max() - pts[:, 0].min()) + pad * 2
            span_z = float(pts[:, 2].max() - pts[:, 2].min()) + pad * 2
            floor_size = max(span_x, span_z, 6.0)
            n_cells = max(12, int(floor_size * 2))
            fv, ft, fc = _make_checkerboard(
                size=floor_size, n=n_cells, y=self._floor_y, cx=cx, cz=cz
            )
            rr.log(
                "world/floor",
                rr.Mesh3D(vertex_positions=fv, triangle_indices=ft, vertex_colors=fc),
                static=True,
            )

        print(f"Logging {T} frames to rerun timeline…")
        for t in range(T):
            rr.set_time("frame", sequence=t)
            rr.set_time("time", duration=t / fps)
            rp = root_positions[t] if root_positions is not None else None
            joints, vertices = self._forward(poses[t], rp)
            self._log_frame(joints, vertices, rp)
        print("Done. Use the rerun timeline scrubber to replay the sequence.")

    def save_animation(
        self,
        poses: torch.Tensor,
        root_positions: Optional[torch.Tensor] = None,
        filepath: str = "animation.rrd",
        fps: float = 30.0,
    ) -> None:
        """Render a pose sequence and save to a ``.rrd`` file.

        Replay with: ``rerun <filepath>``

        Args:
            poses:          (T, 24, 3, 3) local joint rotation matrices.
            root_positions: Optional (T, 3) root translations per frame.
            filepath:       Output ``.rrd`` file path.
            fps:            Timeline frame rate.
        """
        if not filepath.endswith(".rrd"):
            filepath = filepath.rsplit(".", 1)[0] + ".rrd"
        poses, root_positions = _validate_sequence(poses, root_positions)
        T = poses.shape[0]

        # Direct the recording stream to a file instead of the viewer
        rr.save(filepath)

        print(f"Recording {T} frames to {filepath}…")
        # Re-log static elements into the file recording
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        if root_positions is not None:
            pts = root_positions.cpu().numpy().astype(np.float32)
            pts_floor = pts.copy()
            pts_floor[:, 1] = self._floor_y
            rr.log(
                "world/trajectory",
                rr.LineStrips3D([pts_floor], colors=[_to_rgba8(_C_TRAJ)], radii=0.006),
                static=True,
            )
            pad = 2.0
            cx = float((pts[:, 0].min() + pts[:, 0].max()) / 2)
            cz = float((pts[:, 2].min() + pts[:, 2].max()) / 2)
            floor_size = max(
                float(pts[:, 0].max() - pts[:, 0].min()) + pad * 2,
                float(pts[:, 2].max() - pts[:, 2].min()) + pad * 2,
                6.0,
            )
            n_cells = max(12, int(floor_size * 2))
            verts, tris, colors = _make_checkerboard(
                size=floor_size, n=n_cells, y=self._floor_y, cx=cx, cz=cz
            )
        else:
            verts, tris, colors = _make_checkerboard(size=10.0, n=20, y=self._floor_y)
        rr.log(
            "world/floor",
            rr.Mesh3D(
                vertex_positions=verts, triangle_indices=tris, vertex_colors=colors
            ),
            static=True,
        )

        for t in range(T):
            rr.set_time("frame", sequence=t)
            rr.set_time("time", duration=t / fps)
            rp = root_positions[t] if root_positions is not None else None
            joints, vertices = self._forward(poses[t], rp)
            self._log_frame(joints, vertices, rp)

        print(f"Saved: {filepath}  (replay with: rerun {filepath})")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _forward(
        self,
        pose: torch.Tensor,
        root_position: Optional[torch.Tensor],
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        with torch.no_grad():
            _, pos, verts = self._smpl.forward_kinematics(
                pose, root_position=root_position, return_mesh=self._show_mesh
            )
        joints = pos.cpu().numpy().astype(np.float32)
        vertices = verts.cpu().numpy().astype(np.float32) if verts is not None else None
        return joints, vertices

    def _log_frame(
        self,
        joints: np.ndarray,
        vertices: Optional[np.ndarray],
        root_position: Optional[torch.Tensor],
    ) -> None:
        if self._show_mesh and vertices is not None:
            vn = _vertex_normals(vertices, self._faces)
            body_color = np.tile(_to_rgba8(_C_BODY), (len(vertices), 1)).astype(
                np.uint8
            )
            rr.log(
                "world/body/mesh",
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=self._faces,
                    vertex_normals=vn,
                    vertex_colors=body_color,
                ),
            )

        if self._show_skeleton:
            radii = np.where(np.arange(N_JOINTS) == 0, 0.038, 0.022).astype(np.float32)
            joint_colors = np.array(
                [
                    _to_rgba8(_C_ROOT) if j == 0 else _to_rgba8(_C_JOINT)
                    for j in range(N_JOINTS)
                ],
                dtype=np.uint8,
            )
            rr.log(
                "world/skeleton/joints",
                rr.Points3D(joints, radii=radii, colors=joint_colors),
            )

            bone_strips = [[joints[p], joints[c]] for p, c in self._edges]
            rr.log(
                "world/skeleton/bones",
                rr.LineStrips3D(
                    bone_strips,
                    colors=[_to_rgba8(_C_BONE)] * len(bone_strips),
                    radii=0.010,
                ),
            )

        if root_position is not None:
            rp = root_position.cpu().numpy().astype(np.float32)
            rr.log(
                "world/root",
                rr.Points3D([rp], radii=0.045, colors=[_to_rgba8(_C_ROOT)]),
            )
