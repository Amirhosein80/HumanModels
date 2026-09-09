"""
Body Jacobian Demo
==================
Demonstrates computing the Featherstone body Jacobian for the SMPL body model.

The body Jacobian J_i maps joint velocities dq (72-vector for SMPL's 24 ball
joints × 3 DOF) to the 6D spatial velocity of body i:

    v_i = J_i @ dq       (6,) = (6, 72) @ (72,)

Spatial vectors follow Featherstone's Plücker convention:
    v = [ω_x, ω_y, ω_z,  v_x, v_y, v_z]

Run from the project root:
    python examples/jacobian_demo.py
"""

import math

import torch

from hmodel.bmodels.smpl import SMPL
from hmodel.core.jacobian import body_jacobian, motion_subspace

MODEL_PATH = "./models"
GENDER = "male"

# SMPL joint names (for readability)
JOINT_NAMES = [
    "pelvis",
    "l_hip",
    "r_hip",
    "spine1",
    "l_knee",
    "r_knee",
    "spine2",
    "l_ankle",
    "r_ankle",
    "spine3",
    "l_foot",
    "r_foot",
    "neck",
    "l_collar",
    "r_collar",
    "head",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hand",
    "r_hand",
]

J = 24
N_DOF = J * 3  # 72 total DOF (24 ball joints × 3)


def setup_smpl_jacobian(smpl: SMPL, dtype=torch.float64):
    """Build the fixed inputs needed by body_jacobian for SMPL."""
    # Parents: convert kintree to Featherstone convention (parents[0] = -1)
    parents = smpl.kintree[0].tolist()
    parents[0] = -1

    # T-pose bone offsets (fixed anatomical offsets, expressed in parent frame)
    zero_joints, _ = smpl.get_zero_state()
    r = smpl.ik.joint_to_bone(zero_joints).to(dtype)  # (24, 3)

    # Motion subspace: ball joint for every SMPL joint
    S = torch.stack(
        [motion_subspace("ball", dtype=dtype) for _ in range(J)]
    )  # (24, 6, 3)

    return parents, r, S


def demo1_tpose_jacobian(smpl: SMPL) -> None:
    """Compute body Jacobian at T-pose and inspect a few bodies."""
    print("\n── Demo 1: T-pose body Jacobian ──")

    parents, r, S = setup_smpl_jacobian(smpl)
    pose = torch.eye(3, dtype=torch.float64).expand(J, 3, 3).clone()

    Jac = body_jacobian(pose, r, parents, S)  # (24, 6, 72)
    print(f"  Jacobian shape: {Jac.shape}  (body, spatial_dim, dof)")

    # Count non-zero columns per body (= number of ancestors × 3 DOF)
    nonzero_cols = (Jac.abs().sum(dim=-2) != 0).sum(dim=-1)  # (24,)
    for body in [0, 4, 15, 22]:
        print(
            f"  {JOINT_NAMES[body]:12s} (joint {body:2d}): "
            f"{nonzero_cols[body].item():3d} / {N_DOF} non-zero columns"
        )


def demo2_spatial_velocity(smpl: SMPL) -> None:
    """Compute the spatial velocity of several bodies for a known joint motion."""
    print("\n── Demo 2: Spatial velocity from joint velocities ──")

    parents, r, S = setup_smpl_jacobian(smpl)
    pose = torch.eye(3, dtype=torch.float64).expand(J, 3, 3).clone()

    Jac = body_jacobian(pose, r, parents, S)  # (24, 6, 72)

    # Joint velocity: spin the spine (joint 3) about Y at 1 rad/s
    dq = torch.zeros(N_DOF, dtype=torch.float64)
    spine_dof_start = 3 * 3  # joint index 3, Y-axis is column 1 of ball subspace
    dq[spine_dof_start + 1] = 1.0  # Y component

    print("  Input: spine1 rotates about Y at 1 rad/s")
    bodies_to_show = [0, 3, 12, 15]  # pelvis, spine1, neck, head
    for body in bodies_to_show:
        v = Jac[body] @ dq  # (6,)
        omega = v[:3]
        vel = v[3:]
        print(
            f"  {JOINT_NAMES[body]:10s}: "
            f"ω=[{omega[0]:.3f}, {omega[1]:.3f}, {omega[2]:.3f}]  "
            f"v=[{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]"
        )

    # Pelvis (not a descendant of spine1) should have zero velocity
    v_pelvis = Jac[0] @ dq
    assert v_pelvis.abs().max() < 1e-10, "Pelvis should be unaffected by spine motion"
    print("  [OK] Pelvis velocity is zero (pelvis is not a descendant of spine1)")


def demo3_batched(smpl: SMPL) -> None:
    """Batched Jacobian computation over multiple poses."""
    print("\n── Demo 3: Batched Jacobian (8 poses) ──")

    parents, r, S = setup_smpl_jacobian(smpl)
    B = 8

    # Walking: hip flexion/extension
    poses = []
    for b in range(B):
        angle = math.pi / 6 * (b / B)  # 0 .. 30°
        c, s = math.cos(angle), math.sin(angle)
        Rx = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float64)
        pose = torch.eye(3, dtype=torch.float64).expand(J, 3, 3).clone()
        pose[1] = Rx  # left hip
        pose[2] = Rx.T  # right hip
        poses.append(pose)

    E_batch = torch.stack(poses)  # (8, 24, 3, 3)
    r_batch = (
        r.unsqueeze(0).expand(B, J, 3).clone()
    )  # bone offsets are pose-independent

    Jac_batch = body_jacobian(E_batch, r_batch, parents, S)  # (8, 24, 6, 72)
    print(f"  Batched Jacobian shape: {Jac_batch.shape}")

    # The head Jacobian should vary across poses
    head_J = Jac_batch[:, 15]  # (8, 6, 72)
    std_across_batch = head_J.std(dim=0).mean().item()
    print(f"  Head Jacobian variation across poses (std): {std_across_batch:.6f}")


def demo4_gradient(smpl: SMPL) -> None:
    """Show that body_jacobian is differentiable w.r.t. the pose."""
    print("\n── Demo 4: Gradient flow through Jacobian ──")

    parents, r, S = setup_smpl_jacobian(smpl)

    pose = torch.eye(3, dtype=torch.float64).expand(J, 3, 3).clone()
    pose.requires_grad_(True)

    Jac = body_jacobian(pose, r, parents, S)  # (24, 6, 72)

    # Scalar loss: norm of the head Jacobian
    loss = Jac[15].norm()
    loss.backward()

    print(f"  Loss (head Jacobian norm): {loss.item():.6f}")
    print(
        f"  Gradient w.r.t. pose[0] (pelvis): max abs = {pose.grad[0].abs().max():.2e}"
    )
    print(
        f"  Gradient w.r.t. pose[15] (head):  max abs = {pose.grad[15].abs().max():.2e}"
    )


def main() -> None:
    print("Loading SMPL model...")
    smpl = SMPL(path=MODEL_PATH, gender=GENDER)
    print("Loaded.")

    demo1_tpose_jacobian(smpl)
    demo2_spatial_velocity(smpl)
    demo3_batched(smpl)
    demo4_gradient(smpl)

    print("\nAll demos complete.")


if __name__ == "__main__":
    main()
