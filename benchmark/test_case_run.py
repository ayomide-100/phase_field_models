

import time

from experiment import case_path, get_case
from metrics import first_nonfinite_chunk

cases = [
    ("euler", 0.0125, 3),     # stable, about 10 s the first time
    ("spectral", 0.25, 3),    # stable, about 1 s the first time
    ("euler", 0.05, 3),       # blows up in chunk 1; no warm repeats expected
]

start = time.perf_counter()
for method, dt, repeats in cases:
    case = get_case(method, dt, n_warm_repeats=repeats)
    print(f"{method} dt={dt:g}")
    print(f"  file: {case_path(method, dt)}")
    print(f"  stable: {case['stable']}")
    print(f"  first non-finite chunk: {first_nonfinite_chunk(case['finite'])}")
    print(f"  final energy: {case['free_energy'][-1]:.8e}")
    print(f"  final mass: {case['mass'][-1]:.8e}")
    print(f"  final field shape/dtype: {case['final_field'].shape} {case['final_field'].dtype}")
    print(f"  elapsed_warm shape: {case['elapsed_warm'].shape}")
    print(f"  deterministic checked/result: "
          f"{case['deterministic_checked']} / {case['deterministic']}")
print(f"Total wall time: {time.perf_counter() - start:.1f} s")