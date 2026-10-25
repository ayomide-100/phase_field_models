from pfm.energy_models import free_energy_model
from metrics import is_stable
from harness import build_manager, run_chunked
from harness import T, chunk_len
import numpy as np
import os


RESULTS_DIR = os.path.join("benchmark", "results", "raw")


def run_case(method, dt, n_warm_repeats=0):
    chunk_steps = round(chunk_len / dt)
    # if abs(chunk_steps - round(chunk_steps)) < 1e-9:
    #     raise ValueError(f"{chunk_steps} is not within {1e-9}")
    num_chunks = round(T / chunk_len)
    manager = build_manager(method, dt)
    chk_run = run_chunked(manager, chunk_steps, num_chunks, verbose=True)

    final_chk_run = (chk_run["final_field"])
    stable = is_stable(chk_run["finite"])
 
    warm_times = []
    all_match = True
    if n_warm_repeats > 0 and stable:
        for _ in range(n_warm_repeats):
            warm = run_chunked(manager, chunk_steps, num_chunks, verbose=False)
            warm_times.append(warm["elapsed"])
            same = np.array_equal(np.asarray(warm["final_field"]), final_chk_run)
            all_match = all_match and bool(same)
    checked = len(warm_times) > 0
    results = {
        "method" : [],
        "dt" : [],
        "T" : [],
        "total_steps" : [],
        "chunk_steps" : [],
        "num_chunks" : [],
        "initial_mass": [],
        "initial_energy": [],
        "final_field": [],
        "elapsed_cold": [],
        "elapsed_warm": [],
        "stable": [],
        "deterministic_checked": [],
        "deterministic": [],
    }

    results["method"] = method
    results["dt"] = float(dt)
    results["T"] = float(T)
    results["chunk_steps"].append(chunk_steps)
    results["num_chunks"].append(num_chunks)
    results["total_steps"] = chunk_steps * num_chunks
    results["initial_mass"] = ((chk_run["initial_mass"]))
    results["initial_energy"].append((chk_run["initial_energy"]))
    results["final_field"]= np.array(final_chk_run) 
    results["elapsed_cold"].append(np.asarray(chk_run["elapsed"], dtype=np.float64))
    results["elapsed_warm"].append(np.asarray(warm_times, dtype=np.float64).reshape(len(warm_times), num_chunks))
    results["stable"].append(bool(stable))
    results["deterministic_checked"].append(checked)
    results["deterministic"].append(bool(checked and all_match))

    
    final_results = results | chk_run
    return final_results




# run = run_case("spectral", 0.25)
# print(run)


 
def case_path(method, dt):
   
    return os.path.join(RESULTS_DIR, f"{method}_dt{dt:g}.npz")
 
 
def save_case(result):
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = case_path(result["method"], result["dt"])
    np.savez_compressed(path, **result)
    return path
 
 
def load_case(method, dt):
   
    with np.load(case_path(method, dt)) as data:
        return {
            key: (data[key].item() if data[key].ndim == 0 else data[key])
            for key in data.files
        }
 
 
def get_case(method, dt, n_warm_repeats=0, force=False, verbose=True):
    
    path = case_path(method, dt)
    needs_run = force or not os.path.exists(path)
    if not needs_run:
        saved = load_case(method, dt)
        if saved["stable"] and saved["elapsed_warm"].shape[0] < n_warm_repeats:
            needs_run = True
    if needs_run:
        result = run_case(method, dt, n_warm_repeats)
        save_case(result)
    return load_case(method, dt)