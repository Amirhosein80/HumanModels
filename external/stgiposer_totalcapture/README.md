# STGIPoser TotalCapture synthesis — quarantined port

Adapted port of `process_total_capture` (`src/preprocess.py`) + helpers
(`src/preprocess_utils.py`) from
[STGIPoser](https://github.com/Amirhosein80/STGIPoser) (MIT license),
themselves derived from [PIP](https://github.com/Xinyu-Yi/PIP) and
[DynaIP](https://github.com/dx118/dynaip). Original copyright belongs to
their respective authors; this port keeps the algorithm intact while
adapting it to HModel conventions.

## Why it lives here

The PIP-style wide-stencil `_syn_acc` was deliberately removed from
`hmodel.syn_imu`. It is preserved **only** inside this `external/` folder so
STGIPoser-based comparisons stay reproducible without re-polluting the main
library. Do not import these operators into `src/hmodel`.

## Pipeline (per sequence)

1. Load `TotalCapture_Real_60FPS/<s>_<action>.pkl` (real `acc`/`ori`, GlobalPose
   channel order) and the AMASS `<action>_poses.npz` (motion + `trans`).
2. Remap channels with `TC_IMU_MASK = [5, 2, 3, 4, 0, 1]` → repo pelvis-first order.
3. Step motion to ~60 Hz, copy joint 37→23, keep 24 joints, fix root/trans with
   `AMASS_ROT`, zero `IG_JOINTS`, trim to equal length.
4. FK → vertex acceleration `vacc = syn_acc(vert[:, V_MASK])`.
5. **Bias-match**: `imu_acc = acc_real + (vacc.mean − acc_real.mean)` — this is
   how upstream reconciles the gravity/bias gap. `imu_ori` stays real.
6. `jvel = syn_vel(...)`, `uwb = syn_uwb(...)`; save upstream-schema npz
   (`imu_acc, imu_ori, pose, trans, grot, jvel, uwb`), consumable by
   STGIPoser `extract_data_seq`.

## Deltas from upstream

- **Mean SMPL shape**: HModel `SMPL` takes no `betas`; subject shape is an
  accepted, reported caveat (not silently fixed).
- Device/dtype-agnostic (upstream hardcodes `"cuda"`); `fps` parameterised
  (upstream hardcodes 60 Hz). At `fps=60` numerics match upstream.
- Explicit `pkl_path`/`pose_npz_path` args — no hardcoded
  `"TotalCapture_Real_60FPS"` path surgery.
- `run_folder` collects skips (missing pose npz, unsupported framerate)
  instead of only printing.

## Usage

```python
from hmodel.bmodels.smpl import SMPL
from external.stgiposer_totalcapture import process_file, run_folder

smpl = SMPL(path="./models", gender="male")
targets = process_file(
    "D:/Projects/MemPoser/data/Total Capture/TotalCapture_Real_60FPS/s1_walking1.pkl",
    "D:/Projects/MemPoser/data/Total Capture/Pose/s1/walking1_poses.npz",
    smpl,
)
# targets["imu_acc"]: bias-matched real acc; targets["imu_ori"]: real ori
```
