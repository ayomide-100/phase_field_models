import numpy as np


def _as_float_array(x):
    """Convert lists, NumPy arrays or JAX arrays to a float64 NumPy array."""
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def first_nonfinite_chunk(finite_flags):
    """Return the 1-based index of the first chunk whose field was not finite.

    Returns None if every chunk was finite. Multiply the result by the chunk
    length (10 time units) to get the blow-up time.
    """
    for index, ok in enumerate(finite_flags, start=1):
        if not ok:
            return index
    return None


def is_stable(finite_flags, max_abs_rho=None, bound=2.0):
    """Return True only if the run stayed finite (and bounded, if requested).

    The finite flags are checked first, on purpose. After a blow-up the
    recorded max|rho| can read -inf, so it must never be trusted on its own.
    The optional bound check is applied only when every chunk was finite.
    """
    if len(finite_flags) == 0:
        raise ValueError("finite_flags is empty")
    if not all(bool(flag) for flag in finite_flags):
        return False
    if max_abs_rho is not None and np.max(_as_float_array(max_abs_rho)) > bound:
        return False
    return True


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def relative_l2(field, reference):
    """Relative L2 error: norm(field - reference) / norm(reference).

    Both fields are flattened first, so any shape works as long as the two
    shapes match. If the field contains NaN or inf the result is NaN or inf,
    so only call this for runs that were stable.
    """
    f = _as_float_array(field)
    r = _as_float_array(reference)
    if f.shape != r.shape:
        raise ValueError(f"shape mismatch: {f.shape} vs {r.shape}")
    denominator = np.linalg.norm(r.ravel())
    if denominator == 0.0:
        raise ValueError("reference has zero norm")
    return float(np.linalg.norm((f - r).ravel()) / denominator)


def relative_energy_error(energy, reference_energy):
    """Relative error of a scalar energy: |E - E_ref| / |E_ref|."""
    if reference_energy == 0.0:
        raise ValueError("reference energy is zero")
    return float(abs(energy - reference_energy) / abs(reference_energy))


def _clean_pairs(dts, errors):
    """Validate (dt, error) pairs and return them sorted by increasing dt."""
    d = _as_float_array(dts)
    e = _as_float_array(errors)
    if d.ndim != 1 or d.shape != e.shape:
        raise ValueError("dts and errors must be 1-D and the same length")
    if d.size < 2:
        raise ValueError("need at least two points")
    if not (np.all(np.isfinite(d)) and np.all(np.isfinite(e))):
        raise ValueError("dts and errors must be finite; remove unstable runs first")
    if np.any(d <= 0) or np.any(e <= 0):
        raise ValueError(
            "dts and errors must be positive; remove the reference run "
            "(its error is zero) and any failed runs first"
        )
    order = np.argsort(d)
    return d[order], e[order]


def observed_order(dts, errors):
    """Observed order of accuracy.

    Least-squares slope of log(error) against log(dt). A first-order method
    gives a value near 1. Pass only the accuracy-sweep points, with the
    reference run and any unstable runs removed.
    """
    d, e = _clean_pairs(dts, errors)
    slope, _intercept = np.polyfit(np.log(d), np.log(e), 1)
    return float(slope)


def local_orders(dts, errors):
    """Slopes between neighbouring points, sorted by increasing dt.

    Returns a list of length len(dts) - 1. Useful to see where the error
    curve stops being a straight line.
    """
    d, e = _clean_pairs(dts, errors)
    return [
        float(np.log(e[i + 1] / e[i]) / np.log(d[i + 1] / d[i]))
        for i in range(len(d) - 1)
    ]


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------

def mass_drift(masses, initial_mass=None):
    """Largest relative deviation of the mass from its initial value.

    initial_mass: pass the mass of the starting field, for example
    float(manager.average_mass(manager.init_field)). If omitted, the first
    recorded mass is used instead, which is the mass after chunk 1.

    Non-finite entries (chunks after a blow-up) are ignored. Returns NaN if
    nothing usable remains.
    """
    m = _as_float_array(masses)
    if m.size == 0:
        raise ValueError("masses is empty")
    m0 = float(m[0]) if initial_mass is None else float(initial_mass)
    if not np.isfinite(m0) or m0 == 0.0:
        return float("nan")
    usable = m[np.isfinite(m)]
    if usable.size == 0:
        return float("nan")
    return float(np.max(np.abs(usable - m0)) / abs(m0))


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def total_run_time(chunk_times):
    """Total wall time of one run: the sum of the per-chunk elapsed times.

    Use the elapsed list from a warm run (compilation already done). The
    first run of a new manager includes compile time in chunk 1.
    """
    t = _as_float_array(chunk_times)
    if t.size == 0:
        raise ValueError("chunk_times is empty")
    return float(np.sum(t))


def best_total_run_time(repeats):
    """Minimum total run time over repeated runs.

    repeats: a list of elapsed-time lists, one per repeated run. The minimum
    is the least noisy estimate of the true cost.
    """
    if len(repeats) == 0:
        raise ValueError("repeats is empty")
    return min(total_run_time(r) for r in repeats)


def time_per_step_us(total_time, total_steps):
    """Wall time per integration step, in microseconds."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    return 1.0e6 * total_time / total_steps