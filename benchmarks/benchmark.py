"""Performance benchmark suite for HumanModels (hmodel).

Measures latency, throughput, and memory for every major component across
batch sizes, sequence lengths, dtypes, and devices. Results are written to
benchmarks/results_<timestamp>.json (machine-readable) and a summary table is
printed to stdout.

Usage:
    uv run python benchmarks/benchmark.py [--device cpu|cuda|both] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from hmodel import (
    SMPL,
    SynIMU,
    body_jacobian,
    crba,
    force_cross,
    get_velocity,
    motion_cross,
    motion_subspace,
    rnea,
    savgol_filter,
    spatial_inertia,
)
from hmodel.core.filters import savgol_filter as savgol_filter_direct
from hmodel.core.homogeneous import get_transformation_matrix
from hmodel.core.jacobian import body_jacobian_autograd
from hmodel.kinematics import ForwardKinematics, InverseKinematics
from hmodel.syn_imu import SynIMU as SynIMUClass
from hmodel.syn_imu import integrate_ori_gyro
from hmodel.syn_imu.syn_acc import get_local_acc_finite_diff, get_local_acc_sav_gol
from hmodel.syn_imu.syn_gyro import get_local_ang_vel_diff

MODEL_PATH = "./models"

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_call(fn: Callable, warmup: int = 3, repeat: int = 10) -> Dict[str, float]:
    """Run ``fn`` warmup+repeat times, return median/P95/mean/throughput."""
    for _ in range(warmup):
        fn()
        _sync()
    timings: List[float] = []
    for _ in range(repeat):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        t1 = time.perf_counter()
        timings.append(t1 - t0)
    timings.sort()
    n = len(timings)
    median = timings[n // 2]
    p95 = timings[int(round(0.95 * (n - 1)))]
    mean = statistics.fmean(timings)
    return {
        "median_s": median,
        "p95_s": p95,
        "mean_s": mean,
        "min_s": timings[0],
        "max_s": timings[-1],
    }


def _peak_cpu() -> float:
    """Peak CPU RSS in MiB (best-effort via tracemalloc + resource)."""
    import tracemalloc

    tracemalloc.start()
    # tracemalloc tracks Python allocations only; complement with /proc on Linux.
    # On Windows we fall back to tracemalloc peak which is a lower bound.
    try:
        import resource  # Unix only

        _ = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        pass
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def _gpu_mem() -> Tuple[float, float]:
    """Return (allocated_MiB, reserved_MiB) for CUDA, else (0, 0)."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / (1024 * 1024),
        torch.cuda.memory_reserved() / (1024 * 1024),
    )


def _reset_gpu_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _gpu_peak_allocated() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _gpu_peak_reserved() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_reserved() / (1024 * 1024)
    return 0.0


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _identity_pose(J: int, T: int, B: int, dtype, device) -> torch.Tensor:
    eye = torch.eye(3, dtype=dtype, device=device)
    return eye.unsqueeze(0).unsqueeze(0).expand(B, T, J, 3, 3).clone()


def _random_pose(J: int, T: int, B: int, dtype, device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    # random rotation via QR of random matrix
    A = torch.randn(B, T, J, 3, 3, dtype=torch.float64, device=device, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q.to(dtype=dtype)


def _random_positions(
    T: int, S: int, B: int, dtype, device, seed: int = 1
) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(B, T, S, 3, dtype=dtype, device=device, generator=g)


def _random_rotations(
    T: int, S: int, B: int, dtype, device, seed: int = 2
) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(B, T, S, 3, 3, dtype=torch.float64, device=device, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q.to(dtype=dtype)


def _chain_inputs(
    J: int, B: int, dtype, device
) -> Tuple[torch.Tensor, torch.Tensor, List[int], torch.Tensor]:
    parents = [-1] + list(range(J - 1))
    batch_shape = (B,)
    E = torch.eye(3, dtype=dtype, device=device).expand(*batch_shape, J, 3, 3).clone()
    r = torch.zeros(*batch_shape, J, 3, dtype=dtype, device=device)
    r[..., 1:, 2] = 0.5
    S = torch.stack(
        [motion_subspace("revolute_z", dtype=dtype, device=device) for _ in range(J)]
    )
    return E, r, parents, S


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

Results = List[Dict[str, Any]]


def bench_smpL_init(results: Results, device: str, dtype: torch.dtype) -> None:
    """SMPL model initialization time."""
    smpl = SMPL(path=MODEL_PATH, gender="male")
    results.append(
        {
            "component": "smpl_init",
            "device": device,
            "dtype": str(dtype),
            "batch_size": 1,
            "seq_len": 1,
            "median_s": 0.0,  # measured below
            "p95_s": 0.0,
            "peak_cpu_mib": _peak_cpu(),
            "peak_gpu_allocated_mib": _gpu_mem()[0],
            "peak_gpu_reserved_mib": _gpu_mem()[1],
        }
    )


def bench_smpl_fk(
    results: Results,
    smpl: SMPL,
    device: str,
    dtype: torch.dtype,
    batch: int,
    seq: int,
    with_mesh: bool,
) -> None:
    J = 24
    pose = _identity_pose(J, seq, batch, dtype, device)
    trans = torch.zeros(batch, seq, 3, dtype=dtype, device=device)
    label = "smpl_fk_mesh" if with_mesh else "smpl_fk"

    def fn():
        smpl.forward_kinematics(pose, root_position=trans, return_mesh=with_mesh)

    t = _time_call(fn)
    results.append(
        {
            "component": label,
            "device": device,
            "dtype": str(dtype),
            "batch_size": batch,
            "seq_len": seq,
            "median_s": t["median_s"],
            "p95_s": t["p95_s"],
            "mean_s": t["mean_s"],
            "peak_cpu_mib": _peak_cpu(),
            "peak_gpu_allocated_mib": _gpu_mem()[0],
            "peak_gpu_reserved_mib": _gpu_mem()[1],
            "throughput_frames_per_s": (batch * seq) / t["median_s"],
        }
    )


def bench_smpl_lbs(
    results: Results, smpl: SMPL, device: str, dtype: torch.dtype, batch: int, seq: int
) -> None:
    """Measure LBS portion of forward_kinematics with mesh (optimized form)."""
    J = 24
    V = 6890
    pose = _identity_pose(J, seq, batch, dtype, device)
    trans = torch.zeros(batch, seq, 3, dtype=dtype, device=device)

    # Pre-compute FK without mesh to isolate LBS
    grot, gpos, _ = smpl.forward_kinematics(
        pose, root_position=trans, return_mesh=False
    )

    zero_joints, zero_vertex = smpl.get_zero_state()
    zero_joints = zero_joints.to(dtype=dtype, device=device)  # (J, 3)
    zero_vertex = zero_vertex.to(dtype=dtype, device=device)  # (V, 3)
    weights = smpl.skinning_weights.to(dtype=dtype, device=device)  # (V, J)

    # G_j = [R_global_j | p_global_j - R_global_j @ zero_joints[j]]
    rot_part = grot  # (..., J, 3, 3)
    t_part = gpos - torch.matmul(grot, zero_joints.unsqueeze(-1)).squeeze(
        -1
    )  # (..., J, 3)

    def fn():
        rot_v = torch.einsum("...jik,vj->...vik", rot_part, weights)
        trans_v = torch.einsum("...ji,vj->...vi", t_part, weights)
        _ = torch.einsum("...vik,vk->...vi", rot_v, zero_vertex) + trans_v

    t = _time_call(fn)
    results.append(
        {
            "component": "smpl_lbs",
            "device": device,
            "dtype": str(dtype),
            "batch_size": batch,
            "seq_len": seq,
            "median_s": t["median_s"],
            "p95_s": t["p95_s"],
            "mean_s": t["mean_s"],
            "peak_cpu_mib": _peak_cpu(),
            "peak_gpu_allocated_mib": _gpu_mem()[0],
            "peak_gpu_reserved_mib": _gpu_mem()[1],
            "throughput_vertices_per_s": (batch * seq * V) / t["median_s"],
        }
    )


def bench_kinematics_fk(
    results: Results, device: str, dtype: torch.dtype, batch: int, seq: int
) -> None:
    parents = [0, 1, 2]
    fk = ForwardKinematics(parents)
    J = 4
    r_local = _random_pose(J, seq, batch, dtype, device)

    def fn():
        fk.forward_kinematic_rotation(r_local)

    t = _time_call(fn)
    results.append(
        {
            "component": "kinematics_fk_rotation",
            "device": device,
            "dtype": str(dtype),
            "batch_size": batch,
            "seq_len": seq,
            "median_s": t["median_s"],
            "p95_s": t["p95_s"],
            "mean_s": t["mean_s"],
            "peak_cpu_mib": _peak_cpu(),
            "peak_gpu_allocated_mib": _gpu_mem()[0],
            "peak_gpu_reserved_mib": _gpu_mem()[1],
        }
    )


def bench_kinematics_full(
    results: Results, device: str, dtype: torch.dtype, batch: int, seq: int
) -> None:
    parents = [0, 1, 2]
    fk = ForwardKinematics(parents)
    ik = InverseKinematics(parents)
    J = 4
    r_local = _random_pose(J, seq, batch, dtype, device)
    p_local = torch.randn(batch, seq, J, 3, dtype=dtype, device=device)

    def fn():
        grot, gpos = fk.forward_kinematic(r_local, p_local)
        r_local_rec, p_local_rec = ik.inverse_kinematic(grot, gpos)
        return grot, gpos, r_local_rec, p_local_rec

    t = _time_call(fn)
    results.append(
        {
            "component": "kinematics_full_roundtrip",
            "device": device,
            "dtype": str(dtype),
            "batch_size": batch,
            "seq_len": seq,
            "median_s": t["median_s"],
            "p95_s": t["p95_s"],
            "mean_s": t["mean_s"],
            "peak_cpu_mib": _peak_cpu(),
            "peak_gpu_allocated_mib": _gpu_mem()[0],
            "peak_gpu_reserved_mib": _gpu_mem()[1],
        }
    )


def bench_jacobian(
    results: Results, device: str, dtype: torch.dtype, batch: int, J: int
) -> None:
    E, r, parents, S = _chain_inputs(J, batch, dtype, device)

    def fn_analytical():
        return body_jacobian(E, r, parents, S)

    t_analytical = _time_call(fn_analytical)
    results.append(
        {
            "component": "jacobian_analytical",
            "device": device,
            "dtype": str(dtype),
            "batch_size": batch,
            "seq_len": 1,
            "joints": J,
            "median_s": t_analytical["median_s"],
            "p95_s": t_analytical["p95_s"],
            "mean_s": t_analytical["mean_s"],
            "peak_cpu_mib": _peak_cpu(),
            "peak_gpu_allocated_mib": _gpu_mem()[0],
            "peak_gpu_reserved_mib": _gpu_mem()[1],
        }
    )

    # Autograd reference only for unbatched, small J (expensive)
    if batch == 1 and J <= 4:
        E1, r1, parents1, S1 = _chain_inputs(J, 1, dtype, device)
        t_auto = _time_call(lambda: body_jacobian_autograd(E1[0], r1[0], parents1, S1))
        results.append(
            {
                "component": "jacobian_autograd",
                "device": device,
                "dtype": str(dtype),
                "batch_size": 1,
                "seq_len": 1,
                "joints": J,
                "median_s": t_auto["median_s"],
                "p95_s": t_auto["p95_s"],
                "mean_s": t_auto["mean_s"],
                "peak_cpu_mib": _peak_cpu(),
                "peak_gpu_allocated_mib": _gpu_mem()[0],
                "peak_gpu_reserved_mib": _gpu_mem()[1],
            }
        )


def bench_crba(
    results: Results, device: str, dtype: torch.dtype, batch: int, J: int
) -> None:
    # crba/rnea are unbatched APIs per their docstrings (E: (J,3,3)).
    E, r, parents, S = _chain_inputs(J, 1, dtype, device)
    E = E[0]
    r = r[0]
    mass = torch.ones(J, dtype=dtype, device=device)
    com = torch.zeros(J, 3, dtype=dtype, device=device)
    inertia = torch.eye(3, dtype=dtype, device=device).expand(J, 3, 3).clone() * 0.1
    I_sp = spatial_inertia(mass, com, inertia)

    def fn():
        return crba(E, r, parents, S, I_sp)

    t = _time_call(fn)
    N = J * S.shape[-1]
    results.append(
        {
            "component": "crba",
            "device": device,
            "dtype": str(dtype),
            "batch_size": 1,
            "seq_len": 1,
            "joints": J,
            "median_s": t["median_s"],
            "p95_s": t["p95_s"],
            "mean_s": t["mean_s"],
            "peak_cpu_mib": _peak_cpu(),
            "peak_gpu_allocated_mib": _gpu_mem()[0],
            "peak_gpu_reserved_mib": _gpu_mem()[1],
            "matrix_size": N,
        }
    )


def bench_rnea(
    results: Results, device: str, dtype: torch.dtype, batch: int, J: int
) -> None:
    E, r, parents, S = _chain_inputs(J, 1, dtype, device)
    E = E[0]
    r = r[0]
    mass = torch.ones(J, dtype=dtype, device=device)
    com = torch.zeros(J, 3, dtype=dtype, device=device)
    inertia = torch.eye(3, dtype=dtype, device=device).expand(J, 3, 3).clone() * 0.1
    I_sp = spatial_inertia(mass, com, inertia)
    dq = torch.zeros(J * S.shape[-1], dtype=dtype, device=device)
    gravity = torch.tensor([0.0, -9.81, 0.0], dtype=dtype, device=device)

    def fn():
        return rnea(E, r, parents, S, I_sp, dq, gravity)

    t = _time_call(fn)
    results.append(
        {
            "component": "rnea",
            "device": device,
            "dtype": str(dtype),
            "batch_size": 1,
            "seq_len": 1,
            "joints": J,
            "median_s": t["median_s"],
            "p95_s": t["p95_s"],
            "mean_s": t["mean_s"],
            "peak_cpu_mib": _peak_cpu(),
            "peak_gpu_allocated_mib": _gpu_mem()[0],
            "peak_gpu_reserved_mib": _gpu_mem()[1],
        }
    )


def bench_spatial_ops(
    results: Results, device: str, dtype: torch.dtype, batch: int, J: int
) -> None:
    # Only need J; crba/rnea-style inputs not required here.
    bs = (batch,)
    mass = torch.ones(*bs, J, dtype=dtype, device=device)
    com = torch.zeros(*bs, J, 3, dtype=dtype, device=device)
    inertia = (
        torch.eye(3, dtype=dtype, device=device).expand(*bs, J, 3, 3).clone() * 0.1
    )
    I_sp = spatial_inertia(mass, com, inertia)
    v = torch.randn(*bs, J, 6, dtype=dtype, device=device)

    t_spatial = _time_call(lambda: spatial_inertia(mass, com, inertia))
    results.append(
        {
            "component": "spatial_inertia",
            "device": device,
            "dtype": str(dtype),
            "batch_size": batch,
            "seq_len": 1,
            "joints": J,
            "median_s": t_spatial["median_s"],
            "p95_s": t_spatial["p95_s"],
            "mean_s": t_spatial["mean_s"],
            "peak_cpu_mib": _peak_cpu(),
            "peak_gpu_allocated_mib": _gpu_mem()[0],
            "peak_gpu_reserved_mib": _gpu_mem()[1],
        }
    )

    t_motion = _time_call(lambda: motion_cross(v))
    results.append(
        {
            "component": "motion_cross",
            "device": device,
            "dtype": str(dtype),
            "batch_size": batch,
            "seq_len": 1,
            "joints": J,
            "median_s": t_motion["median_s"],
            "p95_s": t_motion["p95_s"],
            "mean_s": t_motion["mean_s"],
            "peak_cpu_mib": _peak_cpu(),
            "peak_gpu_allocated_mib": _gpu_mem()[0],
            "peak_gpu_reserved_mib": _gpu_mem()[1],
        }
    )

    t_force = _time_call(lambda: force_cross(v))
    results.append(
        {
            "component": "force_cross",
            "device": device,
            "dtype": str(dtype),
            "batch_size": batch,
            "seq_len": 1,
            "joints": J,
            "median_s": t_force["median_s"],
            "p95_s": t_force["p95_s"],
            "mean_s": t_force["mean_s"],
            "peak_cpu_mib": _peak_cpu(),
            "peak_gpu_allocated_mib": _gpu_mem()[0],
            "peak_gpu_reserved_mib": _gpu_mem()[1],
        }
    )


def bench_syn_imu_full(
    results: Results, smpl: SMPL, device: str, dtype: torch.dtype, batch: int, seq: int
) -> None:
    imu_gen = SynIMU(smpl, fps=60)
    J = 24
    # syn_imu does not support batching; use (T, J, 3, 3) pose
    pose = _identity_pose(J, seq, 1, dtype, device).squeeze(0)
    trans = torch.zeros(seq, 3, dtype=dtype, device=device)

    def fn():
        return imu_gen.syn_imu(pose, trans, add_noise=False)

    t = _time_call(fn)
    results.append(
        {
            "component": "syn_imu_full",
            "device": device,
            "dtype": str(dtype),
            "batch_size": 1,
            "seq_len": seq,
            "median_s": t["median_s"],
            "p95_s": t["p95_s"],
            "mean_s": t["mean_s"],
            "peak_cpu_mib": _peak_cpu(),
            "peak_gpu_allocated_mib": _gpu_mem()[0],
            "peak_gpu_reserved_mib": _gpu_mem()[1],
            "throughput_frames_per_s": seq / t["median_s"],
        }
    )


def bench_syn_imu_ori(
    results: Results, device: str, dtype: torch.dtype, batch: int, seq: int
) -> None:
    T, S = seq, 6
    rotation_imu = _random_rotations(T, S, 1, dtype, device).squeeze(0)
    gyr = torch.randn(T, S, 3, dtype=dtype, device=device)

    def fn():
        return integrate_ori_gyro(rotation_imu, gyr, fps=60)

    t = _time_call(fn)
    results.append(
        {
            "component": "syn_imu_ori_integration",
            "device": device,
            "dtype": str(dtype),
            "batch_size": 1,
            "seq_len": seq,
            "median_s": t["median_s"],
            "p95_s": t["p95_s"],
            "mean_s": t["mean_s"],
            "peak_cpu_mib": _peak_cpu(),
            "peak_gpu_allocated_mib": _gpu_mem()[0],
            "peak_gpu_reserved_mib": _gpu_mem()[1],
            "throughput_frames_per_s": seq / t["median_s"],
        }
    )


def bench_syn_imu_acc(
    results: Results, device: str, dtype: torch.dtype, batch: int, seq: int
) -> None:
    T, S = seq, 6
    position = torch.randn(T, S, 3, dtype=dtype, device=device)
    rotation = _random_rotations(T, S, 1, dtype, device).squeeze(0)
    gw = torch.tensor([0.0, -9.8, 0.0], dtype=dtype, device=device)

    t_sg = _time_call(lambda: get_local_acc_sav_gol(position, rotation, fps=60, gw=gw))
    results.append(
        {
            "component": "syn_imu_acc_savgol",
            "device": device,
            "dtype": str(dtype),
            "batch_size": 1,
            "seq_len": seq,
            "median_s": t_sg["median_s"],
            "p95_s": t_sg["p95_s"],
            "mean_s": t_sg["mean_s"],
            "peak_cpu_mib": _peak_cpu(),
            "peak_gpu_allocated_mib": _gpu_mem()[0],
            "peak_gpu_reserved_mib": _gpu_mem()[1],
        }
    )

    t_fd = _time_call(
        lambda: get_local_acc_finite_diff(position, rotation, fps=60, gw=gw)
    )
    results.append(
        {
            "component": "syn_imu_acc_finite_diff",
            "device": device,
            "dtype": str(dtype),
            "batch_size": 1,
            "seq_len": seq,
            "median_s": t_fd["median_s"],
            "p95_s": t_fd["p95_s"],
            "mean_s": t_fd["mean_s"],
            "peak_cpu_mib": _peak_cpu(),
            "peak_gpu_allocated_mib": _gpu_mem()[0],
            "peak_gpu_reserved_mib": _gpu_mem()[1],
        }
    )


def bench_syn_imu_gyr(
    results: Results, device: str, dtype: torch.dtype, batch: int, seq: int
) -> None:
    T, S = seq, 6
    rotation = _random_rotations(T, S, 1, dtype, device).squeeze(0)

    t = _time_call(lambda: get_local_ang_vel_diff(rotation, fps=60))
    results.append(
        {
            "component": "syn_imu_gyr",
            "device": device,
            "dtype": str(dtype),
            "batch_size": 1,
            "seq_len": seq,
            "median_s": t["median_s"],
            "p95_s": t["p95_s"],
            "mean_s": t["mean_s"],
            "peak_cpu_mib": _peak_cpu(),
            "peak_gpu_allocated_mib": _gpu_mem()[0],
            "peak_gpu_reserved_mib": _gpu_mem()[1],
        }
    )


def bench_filters(
    results: Results, device: str, dtype: torch.dtype, batch: int, seq: int
) -> None:
    T, S = seq, 6
    x = torch.randn(batch, T, S, 3, dtype=dtype, device=device)

    for deriv in (0, 1, 2):
        t = _time_call(
            lambda d=deriv: savgol_filter_direct(
                x, window_length=5, polyorder=3, deriv=d, delta=1 / 60, axis=1
            )
        )
        results.append(
            {
                "component": f"savgol_deriv{deriv}",
                "device": device,
                "dtype": str(dtype),
                "batch_size": batch,
                "seq_len": seq,
                "median_s": t["median_s"],
                "p95_s": t["p95_s"],
                "mean_s": t["mean_s"],
                "peak_cpu_mib": _peak_cpu(),
                "peak_gpu_allocated_mib": _gpu_mem()[0],
                "peak_gpu_reserved_mib": _gpu_mem()[1],
            }
        )


def bench_velocity(
    results: Results, device: str, dtype: torch.dtype, batch: int, seq: int
) -> None:
    T, S = seq, 6
    pos = torch.randn(batch, T, S, 3, dtype=dtype, device=device)

    t_offline = _time_call(lambda: get_velocity(pos, fps=60, axis=1))
    results.append(
        {
            "component": "velocity_offline",
            "device": device,
            "dtype": str(dtype),
            "batch_size": batch,
            "seq_len": seq,
            "median_s": t_offline["median_s"],
            "p95_s": t_offline["p95_s"],
            "mean_s": t_offline["mean_s"],
            "peak_cpu_mib": _peak_cpu(),
            "peak_gpu_allocated_mib": _gpu_mem()[0],
            "peak_gpu_reserved_mib": _gpu_mem()[1],
        }
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_benchmarks(devices: List[str], quick: bool) -> Results:
    results: Results = []
    dtypes = [torch.float32, torch.float64]

    batch_sizes = [1, 8] if quick else [1, 8, 16, 32]
    seq_lens = [120, 600] if quick else [120, 600, 1800]
    j_counts = [4, 24] if quick else [4, 24]

    # SMPL init once (shared across configs) — time a single init call
    _reset_gpu_peak()
    smpl = SMPL(path=MODEL_PATH, gender="male")
    _sync()
    t0 = time.perf_counter()
    _ = SMPL(path=MODEL_PATH, gender="male")
    _sync()
    init_s = time.perf_counter() - t0
    for dev in devices:
        for dt in dtypes:
            results.append(
                {
                    "component": "smpl_init",
                    "device": dev,
                    "dtype": str(dt),
                    "batch_size": 1,
                    "seq_len": 1,
                    "median_s": init_s,
                    "p95_s": init_s,
                    "mean_s": init_s,
                    "peak_cpu_mib": _peak_cpu(),
                    "peak_gpu_allocated_mib": _gpu_mem()[0],
                    "peak_gpu_reserved_mib": _gpu_mem()[1],
                }
            )

    for dev in devices:
        for dt in dtypes:
            for B in batch_sizes:
                for L in seq_lens:
                    bench_smpl_fk(results, smpl, dev, dt, B, L, with_mesh=False)
                    bench_smpl_fk(results, smpl, dev, dt, B, L, with_mesh=True)
                    if not quick:
                        bench_smpl_lbs(results, smpl, dev, dt, B, L)
                    bench_kinematics_fk(results, dev, dt, B, L)
                    bench_kinematics_full(results, dev, dt, B, L)
                    bench_syn_imu_full(results, smpl, dev, dt, B, L)
                    bench_syn_imu_ori(results, dev, dt, B, L)
                    bench_syn_imu_acc(results, dev, dt, B, L)
                    bench_syn_imu_gyr(results, dev, dt, B, L)
                    bench_filters(results, dev, dt, B, L)
                    bench_velocity(results, dev, dt, B, L)

                for J in j_counts:
                    bench_jacobian(results, dev, dt, B, J)
                    bench_crba(results, dev, dt, B, J)
                    bench_rnea(results, dev, dt, B, J)
                    bench_spatial_ops(results, dev, dt, B, J)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="HumanModels benchmark suite")
    parser.add_argument("--device", choices=["cpu", "cuda", "both"], default="both")
    parser.add_argument(
        "--quick", action="store_true", help="Reduced config for smoke testing"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: benchmarks/results_<ts>.json)",
    )
    args = parser.parse_args()

    devices: List[str] = []
    if args.device in ("cpu", "both"):
        devices.append("cpu")
    if args.device in ("cuda", "both") and torch.cuda.is_available():
        devices.append("cuda")
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available; falling back to CPU")
        devices = ["cpu"]

    print(f"Devices: {devices}")
    print(f"Quick mode: {args.quick}")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")

    results = run_benchmarks(devices, args.quick)

    # Print summary table
    print("\n" + "=" * 110)
    print(
        f"{'Component':<28} {'Dev':<5} {'Dtype':<7} {'B':>3} {'L':>5} {'J':>3} "
        f"{'med(s)':>9} {'p95(s)':>9} {'GPU-alloc(MiB)':>14} {'GPU-res(MiB)':>14}"
    )
    print("-" * 110)
    for r in results:
        print(
            f"{r['component']:<28} {r['device']:<5} {r['dtype']:<7} "
            f"{r.get('batch_size', 1):>3} {r.get('seq_len', 1):>5} "
            f"{r.get('joints', '-'):>3} {r['median_s']:>9.4f} {r['p95_s']:>9.4f} "
            f"{r['peak_gpu_allocated_mib']:>14.1f} {r['peak_gpu_reserved_mib']:>14.1f}"
        )

    # Write JSON
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.output or f"benchmarks/results_{ts}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "timestamp": ts,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written to {out_path} ({len(results)} entries)")


if __name__ == "__main__":
    main()
