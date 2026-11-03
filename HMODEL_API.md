# HModel — API Reference & Tutorial

**HumanModels** is a PyTorch library for human body kinematics and dynamics.
It provides SE(3) transforms, forward/inverse kinematics, Featherstone spatial dynamics,
SMPL body model integration, virtual IMU synthesis, and 3D visualization.

All computation is tensor-first, GPU-ready (CUDA), and batch-aware.

---

## Installation

```bash
# Recommended — uv resolves the CUDA 13.2 PyTorch wheels automatically
uv pip install -e /path/to/HModel

# With visualization extras
uv pip install -e /path/to/HModel[viz]

# Standard pip — requires CUDA-enabled torch to be pre-installed
pip install -e /path/to/HModel
```

**CUDA requirement:** `pyproject.toml` pins `torch` to the CUDA 13.2
wheel index (`https://download.pytorch.org/whl/cu132`). Using `uv` picks this up
automatically. With plain `pip`, install a matching torch wheel first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install -e /path/to/HModel
```

After installation the package is importable as `hmodel`.

---

## Package Layout

| Import path | What it contains |
|---|---|
| `hmodel` | Top-level re-exports of the most common symbols |
| `hmodel.bmodels.smpl` | `SMPL` — body model with FK, LBS, physical properties |
| `hmodel.core.homogeneous` | SE(3) 4×4 matrix utilities |
| `hmodel.core.jacobian` | Plücker 6×6 spatial transforms, body Jacobian |
| `hmodel.core.dynamics` | Spatial inertia, motion/force cross-product operators |
| `hmodel.core.filters` | Savitzky-Golay filter (scipy-compatible) |
| `hmodel.dynamics` | CRBA (mass matrix), RNEA (bias forces), contact Jacobian |
| `hmodel.kinematics` | `ForwardKinematics`, `InverseKinematics` for arbitrary trees |
| `hmodel.feature_ext` | Velocity estimation (SG + causal finite-difference) |
| `hmodel.syn_imu` | Virtual IMU synthesis from pose sequences |
| `hmodel.viz` | Rerun-based 3D viewer (requires `pip install hmodel[viz]`) |

---

## Spatial vector convention

All spatial (6D) vectors follow **Featherstone's Plücker convention**:

```
v = [ω]   ← angular (top 3)
    [v]   ← linear  (bottom 3)
```

---

## Precision / dtype support

All tensor operations in HModel propagate `dtype` from their inputs. The following
precisions are supported unless noted otherwise:

| dtype | FK / LBS | Jacobian | CRBA / RNEA | Filters | Notes |
|---|---|---|---|---|---|
| `torch.float64` | ✓ | ✓ | ✓ | ✓ | Recommended for dynamics |
| `torch.float32` | ✓ | ✓ | ✓ | ✓ | Default for most use cases |
| `torch.float16` | ✓ | ✓ | ✓ | ✓ | GPU only for linalg ops |
| `torch.bfloat16` | ✓ | ✓ | ✓ | ✓ | GPU only for linalg ops |

**Known limitations:**

- `SMPL.__init__` always runs `physical_properties()` in `float64` (scipy/trimesh
  requirement). This is a one-time cost and does not affect inference dtype.
- `SMPL.forward_kinematics(pose)` inherits the dtype of `pose`. Stored T-pose tensors
  (`zero_joints`, `zero_vertex`) are cast to `pose.dtype` at call time.
- `torch.linalg.solve` (used in equations-of-motion examples) may not be supported for
  `float16`/`bfloat16` on CPU. Cast to `float32` or `float64` before solving on CPU;
  CUDA supports `float16`/`bfloat16` for `linalg.solve`.

```python
# Mixed-precision FK example
import torch
from hmodel import SMPL

smpl = SMPL("./models", "male")
pose = torch.eye(3, dtype=torch.bfloat16).expand(24, 3, 3).clone()
rot, pos, verts = smpl.forward_kinematics(pose, return_mesh=True)
assert rot.dtype == torch.bfloat16   # dtype propagated
```

---

## SMPL body model (`hmodel.bmodels.smpl`)

### Class: `SMPL`

```python
from hmodel.bmodels.smpl import SMPL
# or: from hmodel import SMPL

smpl = SMPL(path="./models", gender="male")   # gender: "male" | "female"
```

Constructor pre-computes physical properties (~60 s on first run).

#### Attributes

| Attribute | Shape | Description |
|---|---|---|
| `smpl.mass` | `(J,)` | Per-joint mass in kg |
| `smpl.com` | `(J, 3)` | Per-joint CoM offset in body frame |
| `smpl.inertia` | `(J, 3, 3)` | Per-joint inertia tensor at CoM, body frame |
| `smpl.kintree` | `(2, J)` | Kinematic tree table |
| `smpl.parents` | `(J-1,)` | Parent indices (1-indexed, use `kintree[0]` for Featherstone) |
| `smpl.skinning_weights` | `(V, J)` | LBS skinning weights |
| `smpl.faces` | `(F, 3)` | Mesh face indices |

#### Methods

```python
# T-pose joint + vertex positions (relative to pelvis)
zero_joints, zero_verts = smpl.get_zero_state()
# zero_joints: (J, 3),  zero_verts: (V, 3)

# Forward kinematics + optional mesh deformation
rot, pos, verts = smpl.forward_kinematics(
    pose,                          # (..., J, 3, 3) local rotation matrices
    root_position=None,            # (..., 3) optional global root translation
    return_mesh=True,              # if True, compute skinned vertices
)
# rot:   (..., J, 3, 3)  global rotations
# pos:   (..., J, 3)     global joint positions
# verts: (..., V, 3)     deformed mesh vertices  (None if return_mesh=False)

# Physical properties (cached in smpl.mass / smpl.com / smpl.inertia)
mass, com, inertia = smpl.physical_properties(res=0.02)
```

#### Building Featherstone parents list from SMPL

```python
parents = smpl.kintree[0].tolist()
parents[0] = -1   # root has no parent
```

---

## Homogeneous transforms (`hmodel.core.homogeneous`)

```python
from hmodel.core.homogeneous import (
    get_transformation_matrix,
    inverse_transformation_matrix,
    decode_transformation_matrix,
    skewv, vee,
)
# or: from hmodel import get_transformation_matrix, skewv, ...
```

| Function | Signature | Returns |
|---|---|---|
| `get_transformation_matrix` | `(R: (...,3,3), t: (...,3))` | `(..., 4, 4)` SE(3) matrix |
| `inverse_transformation_matrix` | `(T: (...,4,4))` | `(..., 4, 4)` |
| `decode_transformation_matrix` | `(T: (...,4,4))` | `(R: (...,3,3), t: (...,3))` |
| `skewv` | `(v: (...,3))` | `(..., 3, 3)` skew-symmetric matrix |
| `vee` | `(M: (...,3,3))` | `(..., 3)` extracts vector from skew matrix |

---

## Spatial (Plücker) Jacobian (`hmodel.core.jacobian`)

```python
from hmodel.core.jacobian import (
    plucker_transform,
    inverse_plucker_transform,
    motion_subspace,
    ancestor_mask,
    body_jacobian,
    body_jacobian_autograd,
)
# or: from hmodel import body_jacobian, motion_subspace, ...
```

### `plucker_transform(E, r)` → `(..., 6, 6)`

Builds the 6×6 Plücker spatial transform `X = [E, 0; −E·r×, E]`.

- `E: (..., 3, 3)` — rotation matrix, parent → child
- `r: (..., 3)` — translation from parent origin to child origin, in parent frame

### `inverse_plucker_transform(X)` → `(..., 6, 6)`

Analytical inverse (differentiable, no `linalg.inv`).

### `motion_subspace(joint_type, dtype, device)` → `(6, n_f)`

| `joint_type` | `n_f` | Meaning |
|---|---|---|
| `"revolute_x"` / `"_y"` / `"_z"` | 1 | Single-axis revolute |
| `"prismatic_x"` / `"_y"` / `"_z"` | 1 | Single-axis prismatic |
| `"ball"` | 3 | 3-DOF ball joint (angular axes) |

### `ancestor_mask(parents)` → `(J, J)` bool

`mask[i, j]` is `True` if joint j is an ancestor of joint i (including i itself).

### `body_jacobian(E, r, parents, S)` → `(..., J, 6, N)`

Full body Jacobian for every link. `N = J * n_f`.

```python
J = 24
parents = [-1] + list(range(J - 1))
E = torch.eye(3, dtype=torch.float64).expand(J, 3, 3).clone()
r = smpl.ik.joint_to_bone(zero_joints).to(torch.float64)   # (J, 3)
S = torch.stack([motion_subspace("ball", dtype=torch.float64) for _ in range(J)])

Jac = body_jacobian(E, r, parents, S)   # (24, 6, 72)
# v_i = Jac[i] @ dq    spatial velocity of body i  given joint velocities dq
```

Supports batched inputs: `E: (B, J, 3, 3)` → `Jac: (B, J, 6, N)`.

---

## Spatial dynamics primitives (`hmodel.core.dynamics`)

```python
from hmodel.core.dynamics import spatial_inertia, motion_cross, force_cross
# or: from hmodel import spatial_inertia, motion_cross, force_cross
```

### `spatial_inertia(mass, com, inertia)` → `(..., J, 6, 6)`

Featherstone Eq. 2.63 — builds the 6×6 spatial inertia matrix:

```
I = [ I_c − m·c×·c×,   m·c×  ]
    [     −m·c×,        m·I₃  ]
```

- `mass: (..., J)` — mass in kg
- `com: (..., J, 3)` — CoM offset in body frame
- `inertia: (..., J, 3, 3)` — inertia at CoM, body frame

### `motion_cross(v)` → `(..., 6, 6)`

6×6 motion cross-product matrix for `v×`. Satisfies `v× @ w = v × w` for spatial vectors.

### `force_cross(v)` → `(..., 6, 6)`

Force cross-product `v×* = −(v×)ᵀ`.

---

## Dynamics algorithms (`hmodel.dynamics`)

```python
from hmodel.dynamics import crba, rnea, contact_jacobian
# or: from hmodel import crba, rnea, contact_jacobian
```

### `crba(E, r, parents, S, I_spatial)` → `(N, N)`

**Composite Rigid Body Algorithm** — mass matrix M(q).

Computes `M = Σᵢ Jᵢᵀ Iᵢ Jᵢ` (vectorised, no Python loop over body pairs).
Result is symmetric positive-definite.

### `rnea(E, r, parents, S, I_spatial, dq, gravity)` → `(N,)`

**Recursive Newton-Euler Algorithm** — bias forces h(q, q̇).

Returns the joint torques needed to maintain zero acceleration:
gravity + Coriolis + centripetal effects.

- `dq: (N,)` — joint velocities
- `gravity: (3,)` — gravity vector in world frame, e.g. `[0, -9.81, 0]`

### `contact_jacobian(body_jac, body_indices)` → `(..., 3·n_c, N)`

Extracts the linear-velocity (bottom-3) rows of the body Jacobian for selected bodies.

```python
Jc = contact_jacobian(Jac, [7, 8])   # SMPL ankle joints → (6, 72)
```

### Full equation of motion

```python
M   = crba(E, r, parents, S, I_sp)
h   = rnea(E, r, parents, S, I_sp, dq, gravity)
Jc  = contact_jacobian(body_jacobian(E, r, parents, S), contact_bodies)
tau = torch.zeros(N, dtype=torch.float64)
lam = torch.zeros(3 * len(contact_bodies), dtype=torch.float64)

# τ + Jcᵀ λ = M q̈ + h   →   q̈ = M⁻¹(τ + Jcᵀλ − h)
ddq = torch.linalg.solve(M, tau + Jc.T @ lam - h)
```

---

## Kinematics (`hmodel.kinematics`)

```python
from hmodel.kinematics import ForwardKinematics, InverseKinematics
# or: from hmodel import ForwardKinematics, InverseKinematics
```

### `ForwardKinematics(parents)`

`parents`: list of parent indices (0-indexed, length J−1).

```python
fk = ForwardKinematics(parents=smpl.parents.tolist())

# Global rotations from local rotations
global_rot = fk.forward_kinematic_rotation(local_rot)   # (..., J, 3, 3)

# Global positions from bone offsets
global_pos = fk.bone_to_joint(bone_vectors)             # (..., J, 3)

# Both at once
global_rot, global_pos = fk.forward_kinematic(local_rot, local_offsets)
```

### `InverseKinematics(parents)`

```python
ik = InverseKinematics(parents=smpl.parents.tolist())

# Local rotations from global
local_rot = ik.inverse_kinematic_rotation(global_rot)   # (..., J, 3, 3)

# Bone offset vectors from global joint positions
bone_vecs = ik.joint_to_bone(global_pos)                # (..., J, 3)
```

---

## Filters (`hmodel.core.filters`)

```python
from hmodel.core.filters import savgol_filter
# or: from hmodel import savgol_filter
```

### `savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=0)`

Savitzky-Golay filter matching `scipy.signal.savgol_filter` with `mode='interp'`.

```python
x = torch.randn(100, 24, 3, dtype=torch.float64)   # (T, J, 3)
smooth = savgol_filter(x, window_length=5, polyorder=3, deriv=0, axis=0)
vel    = savgol_filter(x, window_length=5, polyorder=3, deriv=1, delta=1/60, axis=0)
acc    = savgol_filter(x, window_length=5, polyorder=3, deriv=2, delta=1/60, axis=0)
```

---

## Velocity estimation (`hmodel.feature_ext`)

```python
from hmodel.feature_ext import get_velocity, get_realtime_velocity
# or: from hmodel import get_velocity, get_realtime_velocity
```

### `get_velocity(position, fps, axis=0, window_length=5, polyorder=3)`

Offline (non-causal) velocity via Savitzky-Golay 1st derivative.

```python
vel = get_velocity(positions, fps=60)   # same shape as positions
```

### `get_realtime_velocity(position0, position_1, position_2, dt)`

Causal (online) velocity via 2nd-order backward finite difference:
`v(t) = [3x(t) − 4x(t−1) + x(t−2)] / (2Δt)`

```python
vel_t = get_realtime_velocity(pos_t, pos_tm1, pos_tm2, dt=1/60)
```

---

## Virtual IMU synthesis (`hmodel.syn_imu`)

```python
from hmodel.syn_imu import SynIMU, imu_calibration
# or: from hmodel import SynIMU
```

### `SynIMU(smpl, fps)`

```python
imu_gen = SynIMU(smpl, fps=60)

data = imu_gen.syn_imu(
    pose,          # (T, J, 3, 3) pose sequence
    trans,         # (T, 3) root translation
    add_noise=True,
)
# data["ACC"]: (T, 6, 3)    accelerometer readings  (m/s²)
# data["GYR"]: (T, 6, 3)    gyroscope readings      (rad/s)
# data["RBW"]: (T, 6, 3, 3) bone rotation in world frame
# data["RLW"]: (T, 6, 3, 3) local-to-world rotation
# data["RLC"]: (T, 6, 3, 3) local calibration rotation
```

### `imu_calibration(acc, gyr, rbw, rlw, rlc, remove_g=True)`

Transforms raw IMU readings into a calibrated sensor frame.

---

## Visualization (`hmodel.viz`)

Requires: `pip install hmodel[viz]`

```python
from hmodel.viz import SMPLVisualizer

viz = SMPLVisualizer(smpl, show_mesh=True, show_skeleton=True)

# Stream single frames (interactive)
viz.show_frame(pose, root_position=torch.zeros(3, dtype=torch.float64), block=False)
for pose_t in stream:
    viz.update(pose_t)

# Log a full sequence to a scrubable timeline
viz.animate(poses, roots, fps=30)   # poses: (T, J, 3, 3), roots: (T, 3)

# Save to .rrd file for replay with `rerun walk.rrd`
viz.save_animation(poses, roots, filepath="walk.rrd", fps=30)
```

---

## Common recipes

### Forward kinematics for one SMPL pose

```python
import torch
from hmodel import SMPL

smpl = SMPL(path="./models", gender="male")
pose = torch.eye(3, dtype=torch.float64).expand(24, 3, 3).clone()   # T-pose

rot, pos, verts = smpl.forward_kinematics(pose, return_mesh=True)
# rot:   (24, 3, 3)
# pos:   (24, 3)
# verts: (6890, 3)
```

### Batched forward kinematics

```python
B = 8
pose_batch = pose.unsqueeze(0).expand(B, 24, 3, 3).clone()
root_batch = torch.zeros(B, 3, dtype=torch.float64)
rot_b, pos_b, verts_b = smpl.forward_kinematics(pose_batch, root_batch, return_mesh=True)
# verts_b: (8, 6890, 3)
```

### Body Jacobian for SMPL

```python
import torch
from hmodel import SMPL, body_jacobian, motion_subspace, spatial_inertia
from hmodel.dynamics import crba, rnea, contact_jacobian

smpl = SMPL(path="./models", gender="male")
dtype = torch.float64

parents = smpl.kintree[0].tolist(); parents[0] = -1
zero_joints, _ = smpl.get_zero_state()
r = smpl.ik.joint_to_bone(zero_joints).to(dtype)
S = torch.stack([motion_subspace("ball", dtype=dtype) for _ in range(24)])
I_sp = spatial_inertia(smpl.mass.to(dtype), smpl.com.to(dtype), smpl.inertia.to(dtype))

pose = torch.eye(3, dtype=dtype).expand(24, 3, 3).clone()
Jac  = body_jacobian(pose, r, parents, S)   # (24, 6, 72)
M    = crba(pose, r, parents, S, I_sp)      # (72, 72)
g    = torch.tensor([0., -9.81, 0.], dtype=dtype)
h    = rnea(pose, r, parents, S, I_sp, torch.zeros(72, dtype=dtype), g)  # (72,)

# Forward dynamics (free fall, no contact)
ddq = torch.linalg.solve(M, -h)
```

### Real-time velocity (streaming / causal)

```python
from hmodel import get_realtime_velocity

dt = 1 / 60
# Inside your frame loop:
vel_t = get_realtime_velocity(pos_t, pos_tm1, pos_tm2, dt=dt)   # (..., 3)
```

---

## Tensor shape conventions

| Symbol | Meaning |
|---|---|
| `B` | Batch size |
| `J` | Number of joints (24 for SMPL) |
| `V` | Number of vertices (6890 for SMPL) |
| `N` | Total DOF = J × n_f (72 for SMPL ball joints) |
| `n_f` | DOF per joint (1 for revolute, 3 for ball) |
| `T` | Time steps |
| `n_c` | Number of contact points |

## Dtype & device rules

- All functions inherit `dtype` and `device` from their input tensors.
- Never hardcode `torch.float32` or `"cpu"` — pass tensors on the right device.
- SMPL model parameters are stored in `torch.float64` by default.
- CUDA: move tensors with `.to(device)` before passing to any function.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smpl.mass = smpl.mass.to(device)
smpl.com  = smpl.com.to(device)
smpl.inertia = smpl.inertia.to(device)
pose = pose.to(device)
```
