"""
SMPL Visualizer Demo  (rerun backend)
=======================================
SMPL coordinate system: X = right,  Y = up,  Z = forward

Run from the project root:

    python examples/visualizer_demo.py [--demo 1|2|3|4]

  Demo 1  Single T-pose frame (blocking — Ctrl-C to exit)
  Demo 2  Walking animation — logged to rerun timeline (scrubable)
  Demo 3  Save walking animation to  examples/walk.rrd
            Replay with:  rerun examples/walk.rrd
  Demo 4  Online / real-time loop   (streams 120 frames then exits)

Change MODEL_PATH / GENDER at the top if needed.
"""

import argparse
import math
import time

import torch

from hmodel.bmodels.smpl import SMPL
from hmodel.viz import SMPLVisualizer

MODEL_PATH = "./models"
GENDER = "male"
N_JOINTS = 24


# ── rotation helpers ──────────────────────────────────────────────────────────


def Rx(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float64)


def Ry(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)


def Rz(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)


def identity_pose() -> torch.Tensor:
    """(24, 3, 3) all-identity → T-pose."""
    return torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(N_JOINTS, 3, 3).clone()


# ── synthetic motion sequences ────────────────────────────────────────────────

# SMPL standard joint indices
L_HIP, R_HIP = 1, 2
L_KNEE, R_KNEE = 4, 5
L_ANKLE, R_ANKLE = 7, 8
SPINE1, SPINE2 = 3, 6
L_SHLDR, R_SHLDR = 16, 17
L_ELBOW, R_ELBOW = 18, 19


def make_walk_sequence(T: int = 90) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic walk: body moves 4 m along Z (forward in SMPL coords).

    Returns:
        poses:          (T, 24, 3, 3)
        root_positions: (T, 3)
    """
    poses, roots = [], []
    for t in range(T):
        ph = 2 * math.pi * t / T  # full gait cycle
        st = math.sin(ph)  # stride phase −1 … +1

        pose = identity_pose()

        # Hips — swing in the XZ (sagittal) plane (X = right, Z = forward)
        pose[L_HIP] = Rx(0.40 * st)
        pose[R_HIP] = Rx(-0.40 * st)

        # Knees — bend forward only (no hyperextension)
        pose[L_KNEE] = Rx(max(0.0, 0.55 * st))
        pose[R_KNEE] = Rx(max(0.0, -0.55 * st))

        # Ankles — slight plantar-flexion
        pose[L_ANKLE] = Rx(-0.20 * max(0.0, st))
        pose[R_ANKLE] = Rx(-0.20 * max(0.0, -st))

        # Spine — subtle lateral sway
        pose[SPINE1] = Rz(0.04 * st)
        pose[SPINE2] = Rz(0.03 * st)

        # Arms — opposite phase to legs
        pose[L_SHLDR] = Rx(-0.30 * st)
        pose[R_SHLDR] = Rx(0.30 * st)
        pose[L_ELBOW] = Rx(0.20 * max(0.0, -st))
        pose[R_ELBOW] = Rx(0.20 * max(0.0, st))

        poses.append(pose)

        # Root moves along +Z (forward), with small vertical bob
        z = t / T * 4.0  # 4 metres forward
        y = 0.025 * abs(math.sin(ph * 2))  # subtle up/down
        roots.append(torch.tensor([0.0, y, z], dtype=torch.float64))

    return torch.stack(poses), torch.stack(roots)


def make_wave_sequence(T: int = 60) -> tuple[torch.Tensor, torch.Tensor]:
    """Static position, waving right arm."""
    L_ARM_jnts = [R_SHLDR, R_ELBOW]  # reusing naming convention
    poses, roots = [], []
    for t in range(T):
        ph = 2 * math.pi * t / T
        pose = identity_pose()
        pose[R_SHLDR] = Rx(-0.6) @ Rz(-0.3 + 0.5 * math.sin(ph))
        pose[R_ELBOW] = Rx(-0.2 + 0.5 * (1 - math.cos(ph)))
        poses.append(pose)
        roots.append(torch.zeros(3, dtype=torch.float64))
    return torch.stack(poses), torch.stack(roots)


# ── demo functions ────────────────────────────────────────────────────────────


def demo1_single_frame(smpl: SMPL) -> None:
    """Show T-pose in a blocking window."""
    print("\n── Demo 1: T-pose (close window to continue) ──")
    viz = SMPLVisualizer(smpl, show_mesh=True, show_skeleton=True)
    pose = identity_pose()
    root = torch.zeros(3, dtype=torch.float64)
    viz.show_frame(pose, root, block=True)


def demo2_animation(smpl: SMPL) -> None:
    """Interactive walking animation (blocking until window is closed)."""
    print("\n── Demo 2: Walking animation (close window to continue) ──")
    viz = SMPLVisualizer(smpl, show_mesh=True, show_skeleton=False)
    poses, roots = make_walk_sequence(T=90)
    viz.animate(poses, roots, fps=30, repeat=True)


def demo3_save_rrd(smpl: SMPL) -> None:
    """Save walking animation to a .rrd file (replay with: rerun examples/walk.rrd)."""
    out = "examples/walk.rrd"
    print(f"\n── Demo 3: Saving {out} ──")
    viz = SMPLVisualizer(smpl, show_mesh=True, show_skeleton=False)
    poses, roots = make_walk_sequence(T=60)
    viz.save_animation(poses, roots, filepath=out, fps=20)


def demo4_online_loop(smpl: SMPL) -> None:
    """Online / real-time loop: open window once, stream 120 frames."""
    print("\n── Demo 4: Online real-time loop (120 frames) ──")
    viz = SMPLVisualizer(smpl, show_mesh=True, show_skeleton=True)

    # Open window with first frame (non-blocking)
    pose0 = identity_pose()
    viz.show_frame(pose0, block=False)

    poses, roots = make_wave_sequence(T=120)
    dt = 1 / 30
    for t in range(len(poses)):
        viz.update(poses[t], roots[t])
        time.sleep(dt)

    print("Online loop finished.")


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SMPL Visualizer Demo")
    parser.add_argument(
        "--demo",
        type=int,
        default=0,
        help="Which demo to run (1-4). Default 0 = run all in sequence.",
    )
    args = parser.parse_args()

    print("Loading SMPL model…")
    smpl = SMPL(path=MODEL_PATH, gender=GENDER)
    print("Loaded.\n")

    demos = {
        1: demo1_single_frame,
        2: demo2_animation,
        3: demo3_save_rrd,
        4: demo4_online_loop,
    }

    if args.demo in demos:
        demos[args.demo](smpl)
    else:
        for fn in demos.values():
            fn(smpl)

    print("\nAll demos complete.")


if __name__ == "__main__":
    main()
