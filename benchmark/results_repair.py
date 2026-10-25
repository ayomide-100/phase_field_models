"""One-off repair of result files saved with list-wrapped values.

Run from the repository root:  python benchmark/repair_results.py

Older versions of run_case saved single values and tables wrapped in an extra
list, for example num_chunks as [25] (shape (1,)) and elapsed_warm with shape
(1, repeats, chunks). The simulation data inside is correct; only the layout
is wrong. This script unwraps it in place, so the runs do not have to be
repeated.

Safe to run more than once: files that are already correct are left alone.
Each file is copied to benchmark/results/raw_backup before it is changed.
"""

import glob
import os
import shutil

import numpy as np

RESULTS_DIR = os.path.join("benchmark", "results", "raw")
BACKUP_DIR = os.path.join("benchmark", "results", "raw_backup")

# Single values that must be plain scalars, not arrays of shape (1,).
SCALAR_KEYS = [
    "dt", "T", "chunk_steps", "num_chunks", "total_steps",
    "initial_mass", "initial_energy",
    "stable", "deterministic_checked", "deterministic",
]
# key -> number of dimensions the repaired array must have
WRAPPED_ARRAYS = {"elapsed_cold": 2, "elapsed_warm": 3, "final_field": 4}


def repair_case(case):
    """Unwrap one loaded case in place. Returns a list of what was changed."""
    changes = []
    for key in SCALAR_KEYS:
        value = case.get(key)
        if isinstance(value, np.ndarray) and value.shape == (1,):
            case[key] = value.item()
            changes.append(f"{key}: (1,) -> scalar")
    for key, wrapped_ndim in WRAPPED_ARRAYS.items():
        value = case.get(key)
        if isinstance(value, np.ndarray) and value.ndim == wrapped_ndim and value.shape[0] == 1:
            case[key] = value[0]
            changes.append(f"{key}: {value.shape} -> {value[0].shape}")
    return changes


def main(results_dir=RESULTS_DIR, backup_dir=BACKUP_DIR):
    paths = sorted(glob.glob(os.path.join(results_dir, "*.npz")))
    if not paths:
        print(f"No .npz files found in {results_dir}")
        return 1
    repaired = 0
    for path in paths:
        with np.load(path) as data:
            case = {
                key: (data[key].item() if data[key].ndim == 0 else data[key])
                for key in data.files
            }
        changes = repair_case(case)
        name = os.path.basename(path)
        if not changes:
            print(f"ok        {name}")
            continue
        os.makedirs(backup_dir, exist_ok=True)
        backup = os.path.join(backup_dir, name)
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
        np.savez_compressed(path, **case)
        repaired += 1
        print(f"repaired  {name}: " + "; ".join(changes))
    print(f"\n{repaired} of {len(paths)} files repaired. Backups are in {backup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())