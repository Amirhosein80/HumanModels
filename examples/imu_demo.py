"""
Synthetic IMU Demo
==================
Demonstrates generating virtual IMU sensor readings from SMPL body poses.

Six IMU sites: pelvis, left knee, right knee, head, left wrist, right wrist.

Run from the project root:

    python examples/imu_demo.py

Change MODEL_PATH / GENDER at the top if needed.
"""

import math

import torch

from hmodel.bmodels.smpl import SMPL
from hmodel.syn_imu import SynIMU, imu_calibration

MODEL_PATH = "./models"
GENDER = "male"
FPS = 60
T = 120  # number of frames

# Joint indices for the demo walk
L_HIP, R_HIP = 1, 2
L_KNEE, R_KNEE = 4, 5
SPINE1 = 3


# ── rotation helpers ──────────────────────────────────────────────────────────


def Rx(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float64)


def identity_pose(n_joints: int = 24) -> torch.Tensor:
    return torch.eye(3, dtype=torch.float64).expand(n_joints, 3, 3).clone()


# ── demo ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Loading SMPL model…")
    smpl = SMPL(path=MODEL_PATH, gender=GENDER)
    imu_gen = SynIMU(smpl, fps=FPS)
    print("Loaded.")

    # ── Demo 1: stationary T-pose (no noise) ──────────────────────────────────
    print("\n── Demo 1: stationary T-pose ──")
    pose_static = identity_pose().unsqueeze(0).expand(T, 24, 3, 3).clone()
    trans_static = torch.zeros(T, 3, dtype=torch.float64)

    data = imu_gen.syn_imu(pose_static, trans_static, add_noise=False)

    print(f"  ACC shape : {data['ACC'].shape}")  # (T, 6, 3)
    print(f"  GYR shape : {data['GYR'].shape}")  # (T, 6, 3)
    print(f"  RBW shape : {data['RBW'].shape}")  # (T, 6, 3, 3)
    print(f"  ACC[30,0] (pelvis, mid-seq): {data['ACC'][30, 0].tolist()}")
    print(f"  GYR[30,0] (pelvis, mid-seq): {data['GYR'][30, 0].tolist()}")

    # For a stationary body: acc ≈ [0, 9.8, 0] (gravity felt upward in local frame)
    # and gyr ≈ [0, 0, 0]

    # ── Demo 2: walking sequence (with noise) ─────────────────────────────────
    print("\n── Demo 2: walking with noise ──")
    poses, roots = [], []
    for t in range(T):
        ph = 2 * math.pi * t / T
        st = math.sin(ph)
        pose = identity_pose()
        pose[L_HIP] = Rx(0.4 * st)
        pose[R_HIP] = Rx(-0.4 * st)
        pose[L_KNEE] = Rx(max(0.0, 0.5 * st))
        pose[R_KNEE] = Rx(max(0.0, -0.5 * st))
        pose[SPINE1] = Rx(0.03 * st)
        poses.append(pose)
        roots.append(torch.tensor([0.0, 0.0, t / T * 2.0], dtype=torch.float64))

    pose_walk = torch.stack(poses)  # (T, 24, 3, 3)
    trans_walk = torch.stack(roots)  # (T, 3)

    data_noisy = imu_gen.syn_imu(pose_walk, trans_walk, add_noise=True)
    print(f"  ACC std (all sensors): {data_noisy['ACC'].std():.4f} m/s²")
    print(f"  GYR std (all sensors): {data_noisy['GYR'].std():.4f} rad/s")

    # ── Demo 3: IMU calibration ───────────────────────────────────────────────
    print("\n── Demo 3: IMU calibration (identity calib) ──")
    calib = imu_calibration(
        acc=data_noisy["ACC"],
        gyro=data_noisy["GYR"],
        rot_bone_world=data_noisy["RBW"],
        rot_lab_world=data_noisy["RLW"],
        rot_lab_calib=data_noisy["RLC"],
        rot_sensor_bone=data_noisy["RSB"],
        remove_g=True,
    )
    print(f"  acc_calib shape  : {calib['acc_calib'].shape}")
    print(f"  gyro_calib shape : {calib['gyro_calib'].shape}")
    print(f"  rot_bone_calib shape: {calib['rot_bone_calib'].shape}")

    print("\nAll demos complete.")


if __name__ == "__main__":
    main()
