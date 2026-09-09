"""Quarantined port of STGIPoser TotalCapture synthesis (see README.md)."""

from .syn_tc import (
    prepare_motion,
    process_file,
    run_folder,
    save_npz,
    step_for_framerate,
    synthesize_tc_sequence,
)
from .synth_ops import syn_acc, syn_uwb, syn_vel

__all__ = [
    "prepare_motion",
    "process_file",
    "run_folder",
    "save_npz",
    "step_for_framerate",
    "syn_acc",
    "syn_uwb",
    "syn_vel",
    "synthesize_tc_sequence",
]
