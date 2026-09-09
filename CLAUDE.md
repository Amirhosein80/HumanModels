# HModel — CLAUDE.md

## Project Overview

**HumanModels** is a PyTorch-based Python library for human body kinematics and dynamics. It provides:
- Forward/inverse kinematics for skeleton hierarchies
- Homogeneous transformation matrix utilities (SE(3))
- SMPL body model integration with skinning
- Physical properties: mass distribution, center-of-mass, inertia tensors

All computation is tensor-first, GPU-ready (CUDA), and batch-aware.

---

## CLAUDE.md Maintenance Rule

After completing any task in this project — bug fix, new feature, refactor, or test — update this file in the **same response** if the task introduced:
- A new module, class, or public function → add to Module Responsibilities + Usage Examples
- A new dependency → add to Dependencies table
- A new pitfall discovered → add to Common Pitfalls
- A new performance pattern → add to Performance & Optimization Rules

Do this in the same response as the code change, not as a follow-up.

---

## Code Style & Conventions

- **Formatter:** Black, 88-char line length, Python 3.13 target — run `black src/` before committing.
- **Import order:** isort with Black profile — run `isort src/` before committing.
- **Python minimum:** 3.10. Use `match`, `|` union types, and structural pattern matching where they clarify intent.
- **Type hints:** Required on all public functions and class methods. Use `torch.Tensor` (not `Tensor`) for clarity. Annotate shapes in docstrings, not in type hints.
- **Docstrings:** One-line summary + Args/Returns with tensor shapes, e.g. `(B, J, 3)` where B=batch, J=joints.
- **No magic numbers:** All constants (joint counts, matrix dims, etc.) must be named.

---

## Performance & Optimization Rules

These apply whenever you write or refactor code in this repo:

### PyTorch
- **Batch-first always.** Every operation must support a leading batch dimension `B`. Never assume a single sample.
- **Avoid Python loops over tensors.** Replace with `torch.einsum`, `torch.bmm`, broadcasting, or `vmap`. If a loop seems unavoidable, flag it with a `# TODO: vectorize` comment and open a task.
- **Prefer in-place ops carefully.** Use in-place (`add_`, `mul_`) only when the tensor is not part of a computation graph that requires its original value. Never use in-place on tensors produced by `get_transformation_matrix` or other autograd-tracked ops.
- **Pin `.device` and `.dtype` propagation.** New tensors must inherit device/dtype from input tensors — never hardcode `torch.float32` or `"cpu"`.
- **Use `torch.compile`-friendly patterns.** Avoid dynamic shapes, data-dependent control flow, and Python-level branching on tensor values inside hot paths.
- **Prefer `torch.linalg` over manual implementations** for matrix ops (inverse, solve, SVD, eigvalsh, etc.).

### LBS / Skinning
- **Use `torch.einsum` for blending joint transforms by skinning weights**, not `tensordot`:
  ```python
  vertex_transformation = torch.einsum(
      "...jkl,vj->...vkl",
      global_transformation,   # (..., J, 4, 4)
      skinning_weights,        # (V, J)
  )                            # (..., V, 4, 4)
  ```
  This handles arbitrary batch dims, is `torch.compile`-friendly, and avoids an intermediate transpose.
- Never loop per-vertex or per-joint in Python.

### Homogeneous coordinates
- **Use `torch.nn.functional.pad` to append a homogeneous coordinate** instead of zero/one-fill + in-place assignment:
  ```python
  point_h = F.pad(points, (0, 1), value=1.0)   # (N, 4) — affine points
  vec_h   = F.pad(vectors, (0, 1), value=0.0)  # (N, 4) — free vectors
  ```
  In-place fill on a non-leaf tensor blocks gradient flow; `F.pad` does not.

### Transformation matrices
- **Build 4×4 matrices with `torch.cat`**, not zero-init + in-place fill — this propagates gradients through the rotation and position inputs:
  ```python
  top    = torch.cat([rotation, position.unsqueeze(-1)], dim=-1)  # (..., 3, 4)
  bottom = torch.zeros(..., 1, 4, dtype=..., device=...)
  bottom[..., 3] = 1.0
  T      = torch.cat([top, bottom], dim=-2)                       # (..., 4, 4)
  ```

### Memory
- Avoid materializing large intermediate tensors unnecessarily; use `einsum` contraction order to minimize peak memory.
- For SMPL vertex operations (6890 vertices), prefer chunked processing when batch size is large.
- **Cache expensive one-time computations in `__init__`** (T-pose state, physics). Do not recompute them on every forward pass.

### Profiling
- If you identify a bottleneck, add a comment `# BOTTLENECK: <reason>` at the site and suggest the fix.
- When asked to optimize, profile with `torch.profiler` or `torch.utils.benchmark.Timer` and include numbers in your response.

---

## Testing Rules

These apply whenever you write tests:

### Framework & Location
- **Framework:** `pytest` with `pytest-cov` for coverage.
- **Location:** `tests/` at the project root, mirroring `src/hmodel/` structure:
  ```
  tests/
  ├── conftest.py              # shared fixtures (device, dtype, batch sizes, SMPL models)
  ├── core/
  │   ├── test_kinematics.py
  │   ├── test_homogeneous.py
  │   └── test_logger.py
  └── bmodels/
      └── test_smpl.py
  ```

### What to Test
- **Every public class, method, and function** must have at least one test.
- **Tensor shape tests:** Verify output shapes for all supported input shapes (single sample, batched, edge-case batch size 1).
- **Device tests:** Parametrize over `["cpu", "cuda"]` using `pytest.mark.skipif(not torch.cuda.is_available(), ...)` for CUDA tests.
- **Dtype tests:** Parametrize over `[torch.float32, torch.float64, torch.float16, torch.bfloat16]`. For `gradcheck` tests, hardcode `float64` — gradcheck requires double precision. For numerical-correctness tests with tight tolerances (`atol < 1e-3`), skip float16/bfloat16.
- **Numerical correctness:** Use `torch.testing.assert_close` (not `==` or `np.allclose`). Set `atol` and `rtol` explicitly.
- **Inverse consistency:** For FK/IK pairs, verify `IK(FK(x)) ≈ x` and `FK(IK(y)) ≈ y`.
- **Gradient flow:** Use `torch.autograd.gradcheck` on differentiable ops to verify correct gradients.
- **Physics invariants:** Mass must be positive; inertia tensors must be symmetric positive-definite.

### Fixtures (conftest.py)
```python
import pytest, torch
from hmodel.bmodels.smpl import SMPL

@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)

@pytest.fixture(
    params=[torch.float32, torch.float64, torch.float16, torch.bfloat16],
    ids=["f32", "f64", "f16", "bf16"],
)
def dtype(request):
    return request.param

@pytest.fixture(params=[1, 4], ids=["B1", "B4"])
def batch_size(request):
    return request.param

# Session-scoped: SMPL.__init__ runs physical_properties (trimesh + scipy),
# which takes ~60 s. Load each model once per session and share across tests.
@pytest.fixture(scope="session")
def smpl_male():
    return SMPL(path="./models", gender="male")

@pytest.fixture(scope="session")
def smpl_female():
    return SMPL(path="./models", gender="female")
```

### Running Tests
```bash
pytest tests/ -v --cov=src/hmodel --cov-report=term-missing
```

---

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `core.homogeneous` | SE(3) matrix primitives — keep pure, no body-model assumptions |
| `core.kinematics` | FK/IK on abstract skeleton trees — body-model agnostic |
| `core.filters` | General signal filters: `savgol_filter` (mode='interp', matches scipy) |
| `core.logger` | Logging only — no computation |
| `bmodels.smpl` | SMPL-specific: loading, LBS, physics — depends on `core.*` |
| `viz.SMPLVisualizer` | rerun-based 3D viewer for SMPL poses and mesh — depends on `bmodels.smpl` |
| `syn_imu.SynIMU` | Virtual IMU sensor synthesis (acc + gyro) from SMPL poses — depends on `bmodels.smpl`. `acc_mode="savgol"|"paper"`, gyro-integration orientations only, returns bone-to-sensor mount `RSB` |
| `syn_imu.imu_calibration` | Transform raw IMU readings into the lab/world frame (GlobalPose convention). Accepts `rot_sensor_bone` mount |
| `syn_imu.get_local_acc_finite_diff` | 2nd-order FD accelerometer (GlobalPose `IMUSimulator` replica); `get_local_acc_sav_gol` is the SG alternative |
| `syn_imu.syn_ori` | Orientation synthesis: `integrate_ori_gyro` (gyro scan) |
| `external.stgiposer_totalcapture` | Quarantined STGIPoser TotalCapture port (PIP wide-stencil `syn_acc` inside) — never import into `src/hmodel` |
| `datasets.totalcapture` | TotalCapture real-data loading: `.sensors` parsing (wxyz→xyzw!), AMASS `prepare_motion` root fix, `calibrate` to bone frame, geodesic error |
| `feature_ext.get_velocity` | Velocity estimation from position sequences via SG 1st derivative |
| `feature_ext.get_realtime_velocity` | Causal per-frame velocity via 2nd-order backward finite difference |
| `core.jacobian` | Featherstone body Jacobian: Plücker transforms, motion subspaces, body Jacobian |
| `core.dynamics` | Spatial dynamics primitives: `spatial_inertia`, `motion_cross`, `force_cross` |
| `dynamics.crba` | Composite Rigid Body Algorithm — generalized inertia matrix M(q) |
| `dynamics.rnea` | Recursive Newton-Euler Algorithm — bias forces h(q, q̇) |
| `dynamics.contact_jacobian` | Linear-velocity sub-Jacobian for contact point constraints |

New body models (e.g., SMPL-X, STAR) go in `bmodels/` and must reuse `core.kinematics`.

---

## Usage Examples

### core.filters

```python
import torch
from hmodel.core.filters import savgol_filter

x = torch.randn(100, 6, 3, dtype=torch.float64)          # (T, sensors, dims)
smooth = savgol_filter(x, window_length=5, polyorder=3,
                       deriv=0, axis=0)                   # smoothing
acc    = savgol_filter(x, window_length=5, polyorder=3,
                       deriv=2, delta=1/60, axis=0)       # 2nd derivative (acceleration)
```

### core.jacobian

```python
import torch
from hmodel.core.jacobian import plucker_transform, motion_subspace, body_jacobian

# 4-joint chain: root → 1 → 2 → 3, all revolute-Z
J = 4
parents = [-1, 0, 1, 2]
E = torch.eye(3, dtype=torch.float64).expand(J, 3, 3).clone()   # local rotations
r = torch.zeros(J, 3, dtype=torch.float64)                       # local offsets
r[1:, 2] = 0.5                                                   # 0.5 m along Z each

S = torch.stack([
    motion_subspace("revolute_z", dtype=torch.float64)
    for _ in range(J)
])  # (J, 6, 1)

Jac = body_jacobian(E, r, parents, S)  # (J, 6, J*1) = (4, 6, 4)
# Jac[i] is the 6×4 body Jacobian of body i: v_i = Jac[i] @ dq

# Batched: E/r can have leading batch dims
E_batch = E.unsqueeze(0).expand(8, J, 3, 3).clone()
r_batch = r.unsqueeze(0).expand(8, J, 3).clone()
Jac_batch = body_jacobian(E_batch, r_batch, parents, S)  # (8, 4, 6, 4)
```

### core.dynamics + dynamics

```python
import torch
from hmodel.core.dynamics import spatial_inertia, motion_cross, force_cross
from hmodel.core.jacobian import motion_subspace, body_jacobian
from hmodel.dynamics import crba, rnea, contact_jacobian

J = 24   # e.g. SMPL
parents = [-1] + list(range(J - 1))
E = torch.eye(3, dtype=torch.float64).expand(J, 3, 3).clone()
r = torch.zeros(J, 3, dtype=torch.float64)
S = torch.stack([motion_subspace("ball", dtype=torch.float64) for _ in range(J)])  # (J, 6, 3)

# Build 6×6 spatial inertias from SMPL physical properties
I_sp = spatial_inertia(smpl.mass, smpl.com, smpl.inertia)  # (J, 6, 6)

# Mass matrix M(q)  →  (N, N)  where N = J * 3 = 72
M = crba(E, r, parents, S, I_sp)

# Bias forces h(q, dq)  →  (N,)
dq = torch.zeros(J * 3, dtype=torch.float64)
g  = torch.tensor([0., -9.81, 0.], dtype=torch.float64)
h  = rnea(E, r, parents, S, I_sp, dq, g)

# Contact Jacobian for feet (SMPL joints 7=l_ankle, 8=r_ankle)
Jac = body_jacobian(E, r, parents, S)          # (J, 6, N)
Jc  = contact_jacobian(Jac, [7, 8])            # (6, N)   3 rows per contact point

# Equation of motion (with zero contact forces):
#   tau = M @ ddq + h   →   ddq = M⁻¹ @ (tau - h)
tau = torch.zeros(J * 3, dtype=torch.float64)
ddq = torch.linalg.solve(M, tau - h)
```

### feature_ext.get_velocity / get_realtime_velocity

```python
import torch
from hmodel.feature_ext import get_velocity, get_realtime_velocity

positions = torch.randn(120, 24, 3, dtype=torch.float64)  # (T, joints, xyz)

# Offline: non-causal SG filter (uses future samples)
vel = get_velocity(positions, fps=60, axis=0)              # (T, 24, 3)

# Online / streaming: causal 2nd-order backward difference
# pass the three most recent frames each tick
dt = 1 / 60
vel_t = get_realtime_velocity(
    positions[-1], positions[-2], positions[-3], dt=dt
)  # (..., 3) — same shape as one frame
```

### core.homogeneous

```python
import torch
from hmodel.core.homogeneous import (
    get_transformation_matrix,
    inverse_transformation_matrix,
    skewv,
)

R = torch.eye(3).unsqueeze(0)           # (1, 3, 3)
t = torch.tensor([[0.0, 1.0, 0.0]])     # (1, 3)
T = get_transformation_matrix(R, t)     # (1, 4, 4)
T_inv = inverse_transformation_matrix(T)

v = torch.tensor([1.0, 0.0, 0.0])
S = skewv(v)                            # (3, 3) skew-symmetric
```

### core.kinematics

```python
import torch
from hmodel.kinematics.kinematics import forward_kinematics, inverse_kinematics

# parents[0] = -1 (root has no parent), shape (J,)
parents = torch.tensor([-1, 0, 1, 2])
bone_lengths = torch.ones(4)  # (J,)
local_rot = torch.eye(3).expand(4, 3, 3)  # (J, 3, 3) — identity = straight chain

global_rot, global_pos = forward_kinematics(local_rot, bone_lengths, parents)
# global_rot: (J, 3, 3), global_pos: (J, 3)

local_rot_back = inverse_kinematics(global_rot, parents)
# roundtrip: local_rot_back ≈ local_rot
```

### bmodels.smpl

```python
import torch
from hmodel.bmodels.smpl import SMPL

smpl = SMPL(path="./models", gender="male")

# T-pose
pose = torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(24, 3, 3).clone()

# Single sample
rot, pos, verts = smpl.forward_kinematics(pose, return_mesh=True)
# rot: (24, 3, 3), pos: (24, 3), verts: (6890, 3)

# Batched
pose_batch = pose.unsqueeze(0).expand(8, 24, 3, 3)   # (B=8, 24, 3, 3)
root_batch = torch.zeros(8, 3, dtype=torch.float64)
rot_b, pos_b, verts_b = smpl.forward_kinematics(
    pose_batch, root_position=root_batch, return_mesh=True
)
# verts_b: (8, 6890, 3)

# Physical properties
zero_joints, zero_verts = smpl.get_zero_state()
mass, com, inertia = smpl.mass, smpl.com, smpl.inertia
# mass: (24,), com: (24, 3), inertia: (24, 3, 3) symmetric PSD
```

### syn_imu.SynIMU

```python
import torch
from hmodel.bmodels.smpl import SMPL
from hmodel.syn_imu import SynIMU, imu_calibration

smpl = SMPL(path="./models", gender="male")
imu_gen = SynIMU(smpl, fps=60)

T = 120
pose  = torch.eye(3, dtype=torch.float64).expand(24, 3, 3).unsqueeze(0).expand(T, 24, 3, 3).clone()
trans = torch.zeros(T, 3, dtype=torch.float64)

data = imu_gen.syn_imu(pose, trans, add_noise=True)
# data["ACC"]: (T, 6, 3) sensor-local accelerometer  (m/s²)
# data["GYR"]: (T, 6, 3) sensor-local gyroscope      (rad/s)
# data["RBW"]: (T, 6, 3, 3) bone rotation in world frame
# data["RSB"]: (6, 3, 3) bone-to-sensor mount (eye if add_noise=False)
# acc_mode="paper" uses finite differences (GlobalPose replica);
# acc_mode="savgol" (default) uses the SG 2nd derivative.

# Optional: calibrate into lab/world frame (GlobalPose aM/wM convention)
calib = imu_calibration(
    data["ACC"], data["GYR"],
    data["RBW"], data["RLW"], data["RLC"],
    remove_g=True, rot_sensor_bone=data["RSB"],
)
```


### viz.SMPLVisualizer

```python
import torch
from hmodel.bmodels.smpl import SMPL
from hmodel.viz import SMPLVisualizer

smpl = SMPL(path="./models", gender="male")
viz = SMPLVisualizer(smpl, show_mesh=True, show_skeleton=True)

# Online: open viewer, then stream frames
pose = torch.eye(3, dtype=torch.float64).expand(24, 3, 3)
viz.show_frame(pose, root_position=torch.zeros(3, dtype=torch.float64), block=False)
for pose_t in stream:                  # each pose_t is (24, 3, 3)
    viz.update(pose_t)

# Offline: log sequence to rerun timeline (scrubable in viewer)
poses = torch.stack([pose] * 60)       # (T=60, 24, 3, 3)
roots = torch.zeros(60, 3, dtype=torch.float64)
viz.animate(poses, roots, fps=30)

# Save to .rrd file for later replay
viz.save_animation(poses, roots, filepath="walk.rrd", fps=30)
# replay with: rerun walk.rrd
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | All tensor computation — installed from CUDA 13.2 index (`pytorch-cu132`); use `uv` or pre-install a CUDA wheel |
| `roma` | Robust rotation representations (SO(3)/SE(3)) |
| `scipy` | `LinearNDInterpolator` for mesh/physics interpolation (CPU only) |
| `trimesh` | 3D mesh I/O and processing |
| `numpy` | Bridge for non-torch data (loading NPZ, trimesh interop) |
| `rerun-sdk` | 3D visualization viewer (optional, `pip install hmodel[viz]`) |

Avoid adding new dependencies without discussion. Prefer `torch` native ops over numpy where tensors are already on GPU.

**numpy compatibility:** rerun-sdk requires numpy ≥ 2.0. Do **not** add pyrender as a dependency — it requires numpy < 2.0 and conflicts.

---

## Common Pitfalls to Avoid

- Do **not** pass file-order `wxyz` quaternions directly to `roma` (which uses `xyzw`) — reorder to `[x, y, z, w]` first. A wrong-order decode silently corrupts every orientation (identity quats still pass, so add a non-trivial rotation test).
- Do **not** call `.numpy()` on tensors that may be on CUDA — always `.cpu().numpy()`.
- Do **not** use `torch.tensor(existing_tensor)` — use `.clone().detach()` or `torch.as_tensor()`.
- Do **not** store tensors as class attributes without registering them as buffers if the class is an `nn.Module`.
- Do **not** assume joint count — read it from the loaded model data (`kintree_table`, `J_regressor`).
- Do **not** use `assert` for runtime validation in library code — raise `ValueError` with a clear message.
- Do **not** use `tensordot` for skinning — it breaks on higher-rank batch dims and hides transpose bugs. Use `einsum` instead.
- Do **not** use in-place ops (`[..., :3] = x`) on tensors that participate in the autograd graph — use `torch.cat` or `F.pad` to build them functionally.
- Do **not** call `get_zero_state()` or other expensive pure-numpy/scipy helpers inside a hot loop — cache results in `__init__`.
- Do **not** add `pyrender` back as a dependency — it requires numpy < 2.0 and is incompatible with the rest of the stack.
- Do **not** hardcode `dtype=zero_joints.dtype` or similar when building intermediate tensors inside `forward_kinematics` — always derive dtype/device from the input `pose` tensor so that float16/bfloat16 inputs propagate correctly.
- Do **not** use `torch.linalg.solve` (or other `torch.linalg` decompositions) on float16/bfloat16 tensors on CPU — these ops are unsupported on CPU for half precision. Cast to float32/float64 first, or run on CUDA.
