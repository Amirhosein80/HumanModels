# AGENT.md — Project Memory (HModel syn IMU)

Source of truth for code style/tests is `CLAUDE.md`. This file stores
facts and decisions so future sessions don't re-derive them.

## What this repo is
- PyTorch SMPL kinematics + synthetic IMU (`hmodel.syn_imu`), GlobalPose replica.
- Batch-first, GPU-ready, `torch.compile`-friendly patterns.

## Canonical conventions (do not change silently)
- Sensor order pelvis-first: `Pelvis, L_LowLeg, R_LowLeg, Head, L_LowArm, R_LowArm`.
- `_JOINT_MASK = [0, 4, 5, 15, 18, 19]`, `_VERTEX_MASK = [3021, 1176, 4662, 411, 1961, 5424]`
  (pairing matches GlobalPose pelvis-last `v_imu/j_imu`, only order differs).
- Y-up, gravity `gw = (0, -9.8, 0)`. ACC/GYR are sensor-local; `RBW` is
  bone-in-world; `RLW/RLC = eye` (lab == world); `RSB = eye` when `add_noise=False`.

## Syn modes (PIP removed 2026-09-09)
- `acc_mode="savgol"` (default): 5-pt cubic SG 2nd derivative.
- `acc_mode="paper"`: 2nd-order finite differences (GlobalPose `IMUSimulator` replica).
- Orientation: gyro integration only (`integrate_ori_gyro` + `add_ori_noise` +
  `unmount_to_bone`). No direct/PIP orientation path.
- Removed: `get_world_acc_pip`, `get_ori_direct`, `acc_mode="pip"`, `ori_mode="pip"`.

## TotalCapture raw data
- Root: `D:/Projects/MemPoser/data/Total Capture/`.
- `IMU data/<s>/<action>_Xsens_AuxFields.sensors`: cols `name, quat wxyz(4),
  acc(3), gyro(3), mag(3)`. File sensor order is Xsens order; use
  `FILE_IDX = [2, 9, 10, 0, 5, 6]` to reach repo order.
- `Pose/<s>/<action>_poses.npz`: AMASS fit, `poses (T,156)`, `trans (T,3)`,
  `mocap_framerate == 60`.
- `IMU calib/<s>/<s>_<action>_calib_imu_{bone,ref}.txt`: file quats are xyzw;
  apply `BONE_FIX = [[-1,0,0],[0,0,-1],[0,-1,0]]`,
  `REF_FIX = [[-1,0,0],[0,1,0],[0,0,-1]]` (from GlobalPose `process.py`).
- Real bone orientation: `RMB = RIM^T @ RIS @ RSB`.

## Reference
- `https://github.com/Xinyu-Yi/GlobalPose`: `imu_synthesis.py` (`_syn_imu`,
  `skip_ESKF=True` path), `process.py` (`process_totalcapture`), `test.py`
  (`compare_realimu`: `aM = RIM^T @ RIS @ aS + g`, `wM = RIM^T @ RIS @ wS`).

## Open gap hypotheses (raw Xsens vs syn, ranked)
1. World-frame/gravity misalignment (SMPL Y-up vs Xsens world x REF_FIX).
2. Mount `RSB_real` vs identity in clean syn.
3. Zero shape (`betas=0`) vs real subject.
4. AMASS pose-fit error (DIP fits are closer).
5. Translation source (raw npz `trans` vs Vicon root, units/axes/origin).
6. Acc/gyro operator + HW filter differences; time alignment (`[-n:]` tail vs head).

## Commands
- `pytest tests/syn_imu -v`
- `pytest tests/ -v --cov=src/hmodel --cov-report=term-missing`
- `python examples/imu_demo.py`
- `black src/ && isort src/`

## Decisions log
- 2026-09-09: removed PIP code paths (correctness-first, user request).
- 2026-09-09: notebooks reset to single raw-Xsens gap notebook
  (`notebooks/real_vs_syn_raw.ipynb`); preprocessed-pkl comparison dropped.
- 2026-09-09 cross-subject check (frames 0-600, clean `paper` syn):
  gap generalises. Static pelvis gravity compass (quat-free, stands):
  real `mean(aS) ~ [+9.5,+0.3,+2.5]` (gravity on sensor +X; s5 differs:
  `[-0.4,+9.9,+0.8]`, per-session mounting varies) vs syn `~[-1,+0.3,-9.7]`.
  NOTE: that check's ORI geodesic numbers are invalid — it decoded file
  wxyz quats with roma's xyzw order. Fixed in `hmodel.datasets`.
- 2026-09-09: added quarantined STGIPoser TotalCapture port
  (`external/stgiposer_totalcapture`, adapted `process_total_capture`;
  mean-shape caveat, device-agnostic, explicit paths) + comparison notebook
  (`notebooks/stgiposer_vs_real_60fps.ipynb`, 60FPS pkls) +
  `tests/external/test_stgiposer_tc.py`. PIP stencil lives only in
  `external/`; never import into `src/hmodel`.
- 2026-09-09 real-vs-pkl root cause (s1/walking1, measured): pkl acc is
  gravity-removed (mean|a|~4.3 vs raw ~10.8); pkl ori == root-fixed FK
  (raw AMASS poses sit a uniform ~90° away — the missing AMASS_ROT fix);
  raw RIS/aS carry mount + Xsens frame. Fix: new `hmodel.datasets`
  (`parse_sensors` with wxyz→xyzw, `prepare_motion` root fix, GlobalPose
  `calibrate`) + notebook compares syn (via `imu_calibration`) vs
  calibrated real with geodesic metrics. Result on frames 0-600:
  legacy ORI 121-139° → fixed 5.6-19.9°; ACC RMSE ~1.6-2.2 m/s²
  (savgol marginally better than paper). SynIMU untouched.
