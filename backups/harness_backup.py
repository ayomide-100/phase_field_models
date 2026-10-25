import jax
jax.config.update("jax_enable_x64", True)
import time, toml
import numpy as np
import jax.numpy as jnp
from pfm.manager import SimulationManager
import statistics

euler_toml = "Examples/landau/euler_noautodiff.toml"
spectral_toml = "Examples/landau/spectral_noautodiff.toml"

input_file = "Examples/landau/standard_input.dat"
output_file = "benchmark/output"

T = 250
chunk_len = 10
dt = 0.02

    
def build_manager(method, dt):
    
    if method == "euler":
        config_path = euler_toml
    elif method == "spectral":
        config_path = spectral_toml
    else:
        raise ValueError(f"Unknown method: {method}")

    
    steps_float = T / dt
    if not steps_float.is_integer():
        raise ValueError(f"T / dt must be a whole number; got {T / dt}")

    steps = int(steps_float)

    
    with open(config_path, "r") as f:
        config = toml.load(f)

    
    config["dt"] = dt
    config["steps"] = steps
    config["print_every"] = 10**9
    config["print_trajectory_every"] = 10**9
    config["write_path"] = output_file
    config["seed"] = 1
    config["verbose"] = False
    config["load_from"] = input_file

    return SimulationManager(config)


manager = build_manager("euler", dt=dt)
print(manager.init_field.dtype)
print(manager.init_field.shape)


chunk_steps = round(chunk_len / dt)
num_chunks = int(T / chunk_len)
def run_chunked(manager, chunk_steps, num_chunks):
    rho = manager.init_field

    results = {
        "time": [],
        "free_energy": [],
        "mass": [],
        "max_abs_rho": [],
        "finite": [],
        "elapsed": [],
        "final_field": None,
        "status": "success",
    }


    for i in range(1, num_chunks +1):       
        start_time = time.perf_counter()
        print(start_time)
        result = manager._evolve_n_steps(rho, chunk_steps)
        result.block_until_ready()
        elapsed = time.perf_counter() - start_time
        rho = result
        free_energy = float(manager.average_free_energy(result))
        mass = float(manager.average_mass(result))
        max_abs_rho = float(jnp.max(jnp.abs(rho)))
        finite = bool(jnp.isfinite(rho).all())

        simulation_time = i * chunk_len
        end_time = time.perf_counter()
        print(end_time)
        print("Evolution time: ", end_time - start_time," seconds")

        results["time"].append(simulation_time)
        results["free_energy"].append(free_energy)
        results["mass"].append(mass)
        results["max_abs_rho"].append(max_abs_rho)
        results["finite"].append(finite)
        results["elapsed"].append(elapsed)
        results["final_field"] = rho
        


        print(
            f"Chunk {i:02d} | "
            f"t={simulation_time:.1f} | "
            f"energy={free_energy:.8e} | "
            f"mass={mass:.8e} | "
            f"max|rho|={max_abs_rho:.8e} | "
            f"finite={finite} | "
            f"elapsed={elapsed:.4f}s"

        )
    return results


chunk_run  = run_chunked(manager, chunk_steps, num_chunks)




rho = manager.init_field
warm_up_result = manager._evolve_n_steps(rho, chunk_steps)
warm_up_result.block_until_ready()

timings = []

for i in range(15):
    rho_test = manager.init_field

    start_time = time.perf_counter()
    result_test = manager._evolve_n_steps(rho_test, chunk_steps)
    result_test.block_until_ready()
    elapsed = time.perf_counter() - start_time

    timings.append(elapsed)
    print(f"Repeat {i + 1}: {elapsed:.4f} s")

minimum = min(timings)
median = statistics.median(timings)
maximum = max(timings)


print(f"Minimum: {minimum:.4f} s")
print(f"Median:  {median:.4f} s")
print(f"Maximum: {maximum:.4f} s")


half = len(timings) // 2 
min_first_half = min(timings[:half])

difference_percent = (min_first_half - minimum) / minimum * 100

print(f"Minimum First Half: {min_first_half}")
print(f"Difference Percent: {difference_percent}") 

if difference_percent <= 5: 
    print("Minimum has converged, enough repeats")
else:
    print("Minimum still dropping, add more repeats")
