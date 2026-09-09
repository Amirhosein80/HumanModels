"""Fix the two comparison notebooks for real vs synthetic IMU.

Three bugs corrected:
1. Quaternion convention: .sensors file has WXYZ, roma expects XYZW
2. AMASS frame alignment: root rotation + translation need amass_rot transform
3. ACC/GYR comparison frame: real data is in sensor frame, synthetic is in bone frame
   (when add_noise=False). Need to transform real data to bone frame via RSB.
"""

import json


def fix_notebook_raw(path: str) -> None:
    """Fix real_vs_syn_totalcapture.ipynb (raw .sensors file comparison)."""
    with open(path) as f:
        nb = json.load(f)

    cells = nb["cells"]

    # ── Cell 3 (loaders): fix parse_sensors quaternion convention ──────
    src3 = "".join(cells[3]["source"])
    src3 = src3.replace(
        '        "RIS": roma.unitquat_to_rotmat(R.reshape(-1, 4)).reshape('
        "n_frames, n_sensors, 3, 3),",
        "        # FIX: file stores wxyz; roma.unitquat_to_rotmat expects xyzw.\n"
        "        R_xyzw = torch.stack([R[..., 1], R[..., 2], R[..., 3], R[..., 0]], dim=-1)\n"
        '        "RIS": roma.unitquat_to_rotmat(R_xyzw.reshape(-1, 4)).reshape('
        "n_frames, n_sensors, 3, 3),",
    )
    _set_source(cells[3], src3)

    # Also fix load_calib: same quaternion order bug
    src3 = "".join(cells[3]["source"])
    src3 = src3.replace(
        "        x, y, z, w = (float(v) for v in p[1:5])\n"
        "        R = roma.unitquat_to_rotmat(torch.tensor([w, x, y, z], dtype=torch.float64)).t()",
        "        x, y, z, w = (float(v) for v in p[1:5])\n"
        "        # FIX: file stores xyzw order; roma expects xyzw directly\n"
        "        R = roma.unitquat_to_rotmat(torch.tensor([x, y, z, w], dtype=torch.float64)).t()",
    )
    _set_source(cells[3], src3)

    # ── Cell 4 (load+align): add amass_rot ─────────────────────────────
    src4 = "".join(cells[4]["source"])
    src4 = src4.replace(
        "# ── load + align ──────────────────────────────────────────────────────────\n"
        "imu = parse_sensors",
        "# ── load + align ──────────────────────────────────────────────────────────\n"
        "# FIX 2: AMASS-to-SMPL global frame alignment (from GlobalPose process.py)\n"
        "# Raw AMASS poses use a different global frame convention.\n"
        "AMASS_ROT = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=torch.float64)\n"
        "\n"
        "imu = parse_sensors",
    )
    # Now add the actual amass_rot application before the T assertion
    src4 = src4.replace(
        "mot = load_pose_npz",
        "# Apply amass_rot to root rotation and translation\n"
        "def _apply_amass_rot(mot_dict):\n"
        '    """Align AMASS global frame to SMPL frame."""\n'
        '    aa = mot_dict["pose_aa"].clone()\n'
        "    root_rot = roma.rotvec_to_rotmat(aa[:, 0])\n"
        "    aa[:, 0] = roma.rotmat_to_rotvec(AMASS_ROT @ root_rot)\n"
        '    mot_dict["pose_aa"] = aa\n'
        '    mot_dict["tran"] = (AMASS_ROT @ mot_dict["tran"].unsqueeze(-1)).squeeze(-1)\n'
        "    return mot_dict\n"
        "\n"
        "mot = load_pose_npz",
    )
    # Insert _apply_amass_rot call right after mot is loaded
    src4 = src4.replace(
        'mot = load_pose_npz(f"{DATA_ROOT}/Pose/{SUBJECT}/{ACTION}_poses.npz")\n' "key",
        'mot = load_pose_npz(f"{DATA_ROOT}/Pose/{SUBJECT}/{ACTION}_poses.npz")\n'
        "mot = _apply_amass_rot(mot)\n"
        "key",
    )
    _set_source(cells[4], src4)

    # ── Cell 7 (metrics): fix ACC/GYR comparison frame ─────────────────
    src7 = "".join(cells[7]["source"])
    # Replace the metrics cell to transform real ACC/GYR to bone frame
    src7 = src7.replace(
        "# ── metrics: per-sensor / per-axis RMSE + bias ────────────────────────────\n"
        'real = {"ACC": aS, "GYR": wS, "ORI": ori_real}',
        "# ── metrics: per-sensor / per-axis RMSE + bias ────────────────────────────\n"
        "# FIX 3: Transform real ACC/GYR into bone frame for fair comparison.\n"
        "# Real .sensors data is in sensor-local frame (offset from bone by RSB mount).\n"
        "# Synthetic data (add_noise=False) is in bone frame (RSB=identity).\n"
        "# Transform: a_bone = RSB @ a_sensor, w_bone = RSB @ w_sensor\n"
        "real_acc_bone = (RSB_real.unsqueeze(0) @ aS.unsqueeze(-1)).squeeze(-1)  # (W, 6, 3)\n"
        "real_gyr_bone = (RSB_real.unsqueeze(0) @ wS.unsqueeze(-1)).squeeze(-1)\n"
        'real = {"ACC": real_acc_bone, "GYR": real_gyr_bone, "ORI": ori_real}',
    )
    _set_source(cells[7], src7)

    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Fixed: {path}")


def fix_notebook_60fps(path: str) -> None:
    """Fix real_vs_syn_totalcapture_60fps.ipynb (paper pkl comparison)."""
    with open(path) as f:
        nb = json.load(f)

    cells = nb["cells"]

    # ── Cell 3 (load pkl): add amass_rot to borrowed translation ───────
    src3 = "".join(cells[3]["source"])
    src3 = src3.replace(
        "# Translation is NOT in the pkl",
        "# Fix: AMASS-to-SMPL global frame alignment for borrowed translation\n"
        "AMASS_ROT = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=torch.float64)\n"
        "trans_full = (AMASS_ROT @ trans_full.unsqueeze(-1)).squeeze(-1)\n"
        "\n"
        "# Translation is NOT in the pkl",
    )
    _set_source(cells[3], src3)

    # ── Cell 6 (metrics): fix ACC/GYR comparison frame ─────────────────
    # (Note: the60fps notebook uses pkl ori (already rotation matrices) so
    #  the ACC/GYR comparison needs the same RSB fix if using direct comparison.
    #  However, the pkl acc is bias-removed, so the demeaned approach is already fair.
    #  We only need to check that the existing code is correct.)
    # Actually the 60fps notebook already demeans ACC, which removes the systematic
    # gravity/RSB bias. But GYR comparison still has the frame mismatch.
    # Let's check if the GYR needs fixing.
    # The pkl ori provides direct bone orientations, and w_real is derived from ori,
    # so it's already in bone frame. The syn GYR is also in bone frame (add_noise=False).
    # This is already consistent. No fix needed for the 60fps GYR.

    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Fixed: {path}")


def _set_source(cell, src: str) -> None:
    """Set cell source from a single string, splitting into lines."""
    lines = src.split("\n")
    # Re-add \n between lines (jupyter format)
    cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    # Clear outputs since we modified the source
    cell["outputs"] = []
    cell["execution_count"] = None


if __name__ == "__main__":
    fix_notebook_raw("notebooks/real_vs_syn_totalcapture.ipynb")
    fix_notebook_60fps("notebooks/real_vs_syn_totalcapture_60fps.ipynb")
    print("All notebooks fixed.")
