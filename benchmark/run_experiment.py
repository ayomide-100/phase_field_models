
import os
import sys
import time

from experiment import case_path, get_case
from metrics import first_nonfinite_chunk

REF_DT = 0.000125
WARM_REPEATS = 4

EXPLICIT_SWEEP = [0.001, 0.002, 0.004, 0.008, 0.0125]
SPECTRAL_SWEEP = [0.001, 0.002, 0.004, 0.008, 0.016, 0.025, 0.05, 0.1, 0.25]
EXPLICIT_PROBES = [0.016, 0.02, 0.025, 0.05]
SPECTRAL_PROBES = [0.5, 1, 2.5, 5, 10]


def run_list(label, cases):
    """cases is a list of (method, dt, warm_repeats)."""
    print(f"\n=== {label}: {len(cases)} cases")
    for method, dt, repeats in cases:
        path = case_path(method, dt)
        before = os.path.getmtime(path) if os.path.exists(path) else None
        start = time.perf_counter()
        case = get_case(method, dt, n_warm_repeats=repeats)
        seconds = time.perf_counter() - start
        after = os.path.getmtime(path)
        status = "cached" if before is not None and after == before else "ran   "
        bad = first_nonfinite_chunk(case["finite"])
        outcome = "stable" if case["stable"] else f"UNSTABLE (first non-finite chunk {bad})"
        print(
            f"  {method:8s} dt={dt:<9g} {status} {seconds:7.1f} s | "
            f"{outcome} | final energy {case['free_energy'][-1]: .6e}"
        )


def probes():
    cases = [("euler", dt, 0) for dt in EXPLICIT_PROBES]
    cases += [("spectral", dt, 0) for dt in SPECTRAL_PROBES]
    run_list("stability probes", cases)


def reference():
    run_list("reference runs", [("euler", REF_DT, 0), ("spectral", REF_DT, 0)])


def sweep():
    cases = [("euler", dt, WARM_REPEATS) for dt in EXPLICIT_SWEEP]
    cases += [("spectral", dt, WARM_REPEATS) for dt in SPECTRAL_SWEEP]
    run_list("accuracy sweeps", cases)


STAGES = {"probes": probes, "reference": reference, "sweep": sweep}


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "all":
        for name in ("probes", "reference", "sweep"):
            STAGES[name]()
    elif stage in STAGES:
        STAGES[stage]()
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())