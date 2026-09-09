"""
Equations of Motion Demo
========================
Demonstrates the articulated-body equation of motion for the SMPL body model:

    τ + J_c(q)ᵀ λ = M(q) q̈ + h(q, q̇)

Components:
    M(q)  — (72, 72) generalized inertia matrix     via CRBA
    h(q, q̇) — (72,) bias forces (gravity + Coriolis) via RNEA
    J_c   — (6, 72) contact Jacobian for ankle joints

Run from the project root:
    python examples/dynamics_demo.py
"""

import torch

from hmodel.bmodels.smpl import SMPL
from hmodel.core.dynamics import spatial_inertia
from hmodel.core.jacobian import body_jacobian, motion_subspace
from hmodel.dynamics import contact_jacobian, crba, rnea

MODEL_PATH = "./models"
GENDER = "male"

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
GRAVITY = torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64)

# Foot / ankle contact bodies (SMPL indices)
CONTACT_BODIES = [7, 8]  # l_ankle, r_ankle


def setup(smpl: SMPL, dtype=torch.float64):
    """Build common inputs for all demos."""
    parents = smpl.kintree[0].tolist()
    parents[0] = -1

    zero_joints, _ = smpl.get_zero_state()
    r = smpl.ik.joint_to_bone(zero_joints).to(dtype)  # (24, 3)

    S = torch.stack(
        [motion_subspace("ball", dtype=dtype) for _ in range(J)]
    )  # (24, 6, 3)

    I_sp = spatial_inertia(
        smpl.mass.to(dtype),
        smpl.com.to(dtype),
        smpl.inertia.to(dtype),
    )  # (24, 6, 6)

    return parents, r, S, I_sp


def demo1_mass_matrix(smpl: SMPL) -> None:
    """CRBA: compute M(q) and verify it is symmetric PSD."""
    print("\n── Demo 1: Mass matrix M(q) via CRBA ──")

    parents, r, S, I_sp = setup(smpl)
    pose = torch.eye(3, dtype=torch.float64).expand(J, 3, 3).clone()

    M = crba(pose, r, parents, S, I_sp)  # (72, 72)
    print(f"  M shape: {tuple(M.shape)}")

    sym_err = (M - M.T).abs().max().item()
    eigs = torch.linalg.eigvalsh(M)
    print(f"  Symmetry error:      {sym_err:.2e}")
    print(f"  Min eigenvalue:      {eigs.min().item():.4f}  (> 0 → PSD ✓)")
    print(f"  Max eigenvalue:      {eigs.max().item():.4f}")
    print(f"  Condition number:    {(eigs.max() / eigs.min()).item():.1f}")


def demo2_bias_forces(smpl: SMPL) -> None:
    """RNEA: h(q, 0) — gravity torques at T-pose."""
    print("\n── Demo 2: Bias forces h(q, q̇=0) via RNEA ──")

    parents, r, S, I_sp = setup(smpl)
    pose = torch.eye(3, dtype=torch.float64).expand(J, 3, 3).clone()
    dq = torch.zeros(N_DOF, dtype=torch.float64)

    h = rnea(pose, r, parents, S, I_sp, dq, GRAVITY)  # (72,)

    print("  Gravity torques per joint (norm of 3-DOF block):")
    for j in [0, 1, 3, 6, 12, 15]:
        tau_j = h[j * 3 : (j + 1) * 3]
        print(
            f"    {JOINT_NAMES[j]:12s} (joint {j:2d}): "
            f"‖τ‖ = {tau_j.norm().item():.4f} N·m"
        )


def demo3_forward_dynamics(smpl: SMPL) -> None:
    """Solve q̈ = M⁻¹(τ − h) with τ=0 (free fall under gravity)."""
    print("\n── Demo 3: Forward dynamics q̈ = M⁻¹(τ − h) ──")

    parents, r, S, I_sp = setup(smpl)
    pose = torch.eye(3, dtype=torch.float64).expand(J, 3, 3).clone()
    dq = torch.zeros(N_DOF, dtype=torch.float64)

    M = crba(pose, r, parents, S, I_sp)
    h = rnea(pose, r, parents, S, I_sp, dq, GRAVITY)
    tau = torch.zeros(N_DOF, dtype=torch.float64)

    ddq = torch.linalg.solve(M, tau - h)  # (72,)

    print(f"  ddq norm: {ddq.norm().item():.4f} rad/s²")
    print(
        "  Root (pelvis) linear acceleration (joints 0-2):",
        [f"{v:.3f}" for v in ddq[:3].tolist()],
    )


def demo4_contact_jacobian(smpl: SMPL) -> None:
    """Contact Jacobian J_c and full EOM residual check."""
    print("\n── Demo 4: Contact Jacobian + EOM residual ──")

    parents, r, S, I_sp = setup(smpl)
    pose = torch.eye(3, dtype=torch.float64).expand(J, 3, 3).clone()
    dq = torch.zeros(N_DOF, dtype=torch.float64)

    Jac = body_jacobian(pose, r, parents, S)  # (24, 6, 72)
    Jc = contact_jacobian(Jac, CONTACT_BODIES)  # (6, 72)
    print(
        f"  Contact Jacobian shape: {tuple(Jc.shape)}  "
        f"({len(CONTACT_BODIES)} bodies × 3 linear rows)"
    )

    # EOM residual: M q̈ + h − τ − Jcᵀ λ = 0
    # Choose a known λ (zero) and τ (zero), solve for q̈
    M = crba(pose, r, parents, S, I_sp)
    h = rnea(pose, r, parents, S, I_sp, dq, GRAVITY)
    tau = torch.zeros(N_DOF, dtype=torch.float64)
    lam = torch.zeros(3 * len(CONTACT_BODIES), dtype=torch.float64)

    ddq = torch.linalg.solve(M, tau + Jc.T @ lam - h)
    residual = M @ ddq + h - tau - Jc.T @ lam
    print(f"  EOM residual ‖M q̈ + h − τ − Jcᵀλ‖: {residual.norm().item():.2e}  (≈ 0 ✓)")


def main() -> None:
    print("Loading SMPL model...")
    smpl = SMPL(path=MODEL_PATH, gender=GENDER)
    print("Loaded.")

    demo1_mass_matrix(smpl)
    demo2_bias_forces(smpl)
    demo3_forward_dynamics(smpl)
    demo4_contact_jacobian(smpl)

    print("\nAll demos complete.")


if __name__ == "__main__":
    main()
