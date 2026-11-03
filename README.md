# HModel — HumanModels

**HumanModels** is a PyTorch library for human body kinematics and dynamics. It provides:

- **Forward / inverse kinematics** for arbitrary skeleton trees
- **Homogeneous transformation matrix** utilities (SE(3))
- **Featherstone spatial dynamics** — body Jacobians, CRBA mass matrix, RNEA bias forces, contact Jacobians
- **SMPL body model** integration with linear blend skinning and physical properties (mass, center of mass, inertia)
- **Synthetic IMU synthesis** — accelerometer / gyroscope signals from pose sequences, with lab-frame calibration
- **Feature extraction** — Savitzky-Golay and causal real-time velocity estimation
- **3D visualization** of SMPL poses and meshes via [rerun](https://rerun.io) (optional)

All computation is tensor-first, batch-first, GPU-ready (CUDA), and `torch.compile`-friendly.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/). PyTorch is resolved from the
CUDA 13.2 wheel index automatically.

```bash
git clone https://github.com/amirhossein-feiz/hmodel
cd hmodel
uv sync                      # runtime + dev dependencies
uv sync --extra viz          # + rerun 3D viewer
```

Editable install with plain `pip` (install a CUDA-enabled torch first):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install -e .
pip install -e ".[viz]"
```

Requires Python ≥ 3.10.

## Quick start

```python
import torch
from hmodel import SMPL, SynIMU, ForwardKinematics

smpl = SMPL(path="./models", gender="male")

# T-pose: (24, 3, 3) local rotation matrices
pose = torch.eye(3, dtype=torch.float64).expand(24, 3, 3).clone()
rot, pos, verts = smpl.forward_kinematics(pose, return_mesh=True)

# Batched: (B, 24, 3, 3) -> (B, 6890, 3)
pose_batch = pose.unsqueeze(0).expand(8, 24, 3, 3).clone()
root = torch.zeros(8, 3, dtype=torch.float64)
_, _, verts_b = smpl.forward_kinematics(pose_batch, root_position=root, return_mesh=True)

# Synthetic IMU from a pose sequence
imu_gen = SynIMU(smpl, fps=60)
data = imu_gen.syn_imu(pose_sequence, translation)   # data["ACC"], data["GYR"], ...
```

More examples live in [`examples/`](examples/), and the full API walkthrough is in
[`HMODEL_API.md`](HMODEL_API.md).

## Development

```bash
uv sync --group dev         # pytest, pytest-cov, black, isort, pre-commit
uv run pytest               # run the test suite
uv run black src/ tests/    # format
uv run isort src/ tests/    # sort imports (Black profile)
uv build                    # build sdist + wheel into dist/
```

Project conventions for agents and contributors are documented in [`AGENTS.md`](AGENTS.md).

## Project layout

```text
src/hmodel/
├── core/          # SE(3) transforms, Jacobians, spatial dynamics, filters, logger
├── kinematics/    # forward/inverse kinematics on abstract skeleton trees
├── dynamics/      # CRBA, RNEA, contact Jacobian
├── bmodels/       # SMPL body model (loading, LBS, physics)
├── syn_imu/       # virtual accelerometer/gyroscope/orientation synthesis
├── feature_ext/   # velocity estimation
└── viz/           # rerun-based SMPL viewer (optional extra)
tests/             # pytest suite mirroring src/hmodel/
examples/          # runnable demos
models/            # SMPL NPZ body models (not shipped in the wheel)
```

## Contributing

1. Create a feature branch and keep changes focused.
2. Follow the style and performance rules in [`AGENTS.md`](AGENTS.md).
3. Add tests under `tests/` for every public function you touch.
4. Run `uv run pytest` and `uv run black src/ tests/` before opening a PR.

## License

MIT — see [LICENSE](LICENSE).
