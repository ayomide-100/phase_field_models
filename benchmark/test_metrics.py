"""Tests for metrics.py. Run directly:  python benchmark/test_metrics.py"""

import math

import numpy as np

from metrics import best_total_run_time, first_nonfinite_chunk, is_stable, local_orders, mass_drift, observed_order, relative_energy_error, relative_l2, time_per_step_us, total_run_time,



def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def raises(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return True
    return False


def test_stability():
    assert first_nonfinite_chunk([True, True, False, False]) == 3
    assert first_nonfinite_chunk([True, True, True]) is None
    assert first_nonfinite_chunk([False]) == 1

    assert is_stable([True, True, True]) is True
    assert is_stable([True, True, False, False]) is False
    assert raises(is_stable, [])

    # Optional bound guard, applied only when everything is finite.
    assert is_stable([True, True], [0.5, 1.5]) is True
    assert is_stable([True, True], [0.5, 2.5]) is False
    # A blown-up run whose max|rho| reads -inf must still be unstable.
    assert is_stable([True, False], [0.5, -np.inf]) is False


def test_relative_l2():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(1, 64, 64))
    assert close(relative_l2(a, a), 0.0)
    assert close(relative_l2(2.0 * a, a), 1.0)
    assert close(relative_l2(1.1 * a, a), 0.1)
    assert raises(relative_l2, a, a[:, :32, :])
    assert raises(relative_l2, a, np.zeros_like(a))
    bad = a.copy()
    bad[0, 0, 0] = np.nan
    assert math.isnan(relative_l2(bad, a))


def test_relative_energy_error():
    assert close(relative_energy_error(-1.1, -1.0), 0.1)
    assert close(relative_energy_error(-1.0, -1.0), 0.0)
    assert raises(relative_energy_error, -1.0, 0.0)


def test_mass_drift():
    assert close(mass_drift([1.0, 1.0, 1.0]), 0.0)
    assert close(mass_drift([1.0, 1.001, 0.999]), 0.001)
    # Non-finite entries (after a blow-up) are ignored.
    assert close(mass_drift([1.0, 1.001, float("nan"), float("nan")]), 0.001)
    # Explicit initial mass.
    assert close(mass_drift([1.01, 1.01], initial_mass=1.0), 0.01)
    # Blown up in the first chunk: undefined, returned as NaN.
    assert math.isnan(mass_drift([float("nan"), float("nan")]))
    assert raises(mass_drift, [])


def test_observed_order():
    dts = [0.001, 0.002, 0.004, 0.008]
    assert close(observed_order(dts, [3.0 * d for d in dts]), 1.0, 1e-9)
    assert close(observed_order(dts, [5.0 * d ** 2 for d in dts]), 2.0, 1e-9)
    # Unsorted input gives the same answer.
    shuffled = [0.004, 0.001, 0.008, 0.002]
    assert close(observed_order(shuffled, [3.0 * d for d in shuffled]), 1.0, 1e-9)
    # Local slopes are all 1 for an exactly first-order error.
    locals_ = local_orders(dts, [3.0 * d for d in dts])
    assert len(locals_) == 3 and all(close(s, 1.0, 1e-9) for s in locals_)
    # Bad inputs are rejected.
    assert raises(observed_order, [0.001], [1e-5])
    assert raises(observed_order, [0.001, 0.002], [0.0, 1e-5])
    assert raises(observed_order, [0.001, 0.002], [float("nan"), 1e-5])
    assert raises(observed_order, [0.001, 0.002], [1e-5])


def test_cost():
    assert close(total_run_time([1.0, 2.0, 3.0]), 6.0)
    assert close(best_total_run_time([[1.0, 2.0], [0.5, 0.7], [3.0, 3.0]]), 1.2)
    assert close(time_per_step_us(1.0, 1_000_000), 1.0)
    assert close(time_per_step_us(0.7, 10_000), 70.0)
    assert raises(total_run_time, [])
    assert raises(best_total_run_time, [])
    assert raises(time_per_step_us, 1.0, 0)


if __name__ == "__main__":
    tests = [
        test_stability,
        test_relative_l2,
        test_relative_energy_error,
        test_mass_drift,
        test_observed_order,
        test_cost,
    ]
    for test in tests:
        test()
        print(f"passed: {test.__name__}")
    print("all tests passed")