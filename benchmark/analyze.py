import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import (
    best_total_run_time,
    first_nonfinite_chunk,
    local_orders,
    observed_order,
    relative_energy_error,
    relative_l2,
    time_per_step_us,
)

RESULTS_DIR = os.path.join("benchmark", "results", "raw")
OUT_DIR = os.path.join("benchmark", "results")
REF_DT = 0.000125

# The timed accuracy sweep. Used for the order fit and the cost curves.
# Keep in sync with run_experiments.py.
SWEEP_DTS = {
    "euler": [0.001, 0.002, 0.004, 0.008, 0.0125],
    "spectral": [0.001, 0.002, 0.004, 0.008, 0.016, 0.025, 0.05, 0.1, 0.25],
}
METHODS = ("euler", "spectral")
LABELS = {"euler": "Explicit Euler", "spectral": "Semi-implicit (spectral)"}
COLORS = {"euler": "tab:red", "spectral": "tab:blue"}

# (row key, table heading)
COLUMNS = [
    ("method", "method"),
    ("dt", "dt"),
    ("steps", "steps"),
    ("stable", "stable"),
    ("blowup_at", "blow-up detected at t"),
    ("rel_l2", "rel. L2 error"),
    ("rel_energy", "rel. energy error"),
    ("runtime_s", "runtime (s)"),
    ("us_per_step", "us per step"),
]


def load_all(results_dir):
    """Load every saved case into a dict keyed by (method, dt)."""
    cases = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.npz"))):
        with np.load(path) as data:
            case = {
                key: (data[key].item() if data[key].ndim == 0 else data[key])
                for key in data.files
            }
        cases[(case["method"], float(case["dt"]))] = case
    return cases


def find_reference(cases, method, ref_dt):
    for (m, dt), case in cases.items():
        if m == method and np.isclose(dt, ref_dt):
            return case
    raise FileNotFoundError(
        f"No reference run for '{method}' at dt={ref_dt:g}. "
        "Run: python benchmark/run_experiments.py reference"
    )


def build_rows(cases, ref_dt):
    """One row per (method, dt), excluding the reference runs themselves."""
    refs = {m: find_reference(cases, m, ref_dt) for m in METHODS}
    rows = []
    for (method, dt), case in sorted(cases.items(), key=lambda item: item[0]):
        if np.isclose(dt, ref_dt):
            continue
        chunk_len = case["T"] / case["num_chunks"]
        first_bad = first_nonfinite_chunk(case["finite"])
        stable = first_bad is None
        ref = refs[method]
        row = {
            "method": method,
            "dt": dt,
            "steps": int(case["total_steps"]),
            "T": float(case["T"]),
            "stable": stable,
            "blowup_at": None if stable else first_bad * chunk_len,
            "rel_l2": np.nan,
            "rel_energy": np.nan,
            "runtime_s": np.nan,
            "us_per_step": np.nan,
        }
        if stable:
            row["rel_l2"] = relative_l2(case["final_field"], ref["final_field"])
            row["rel_energy"] = relative_energy_error(
                case["free_energy"][-1], ref["free_energy"][-1]
            )
            warm = np.asarray(case["elapsed_warm"])
            if warm.ndim == 2 and warm.shape[0] > 0 and warm.shape[1] > 0:
                row["runtime_s"] = best_total_run_time(list(warm))
                row["us_per_step"] = time_per_step_us(row["runtime_s"], row["steps"])
            elif warm.size > 0 or warm.ndim != 2:
                print(f"WARNING: {method} dt={dt:g}: elapsed_warm has shape {warm.shape}, "
                      "expected (repeats, chunks). Runtime skipped; delete this file and re-run the case.")
        rows.append(row)
    return rows


def format_value(key, value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if key == "dt":
        return f"{value:g}"
    if key == "stable":
        return "yes" if value else "no"
    if key in ("rel_l2", "rel_energy"):
        return f"{value:.3e}"
    if key == "runtime_s":
        return f"{value:.3f}"
    if key == "us_per_step":
        return f"{value:.1f}"
    if key == "blowup_at":
        return f"{value:g}"
    return str(value)


def write_tables(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([heading for _, heading in COLUMNS])
        for row in rows:
            writer.writerow(
                ["" if row[k] is None or (isinstance(row[k], float) and np.isnan(row[k]))
                 else row[k] for k, _ in COLUMNS]
            )
    headings = [heading for _, heading in COLUMNS]
    lines = ["| " + " | ".join(headings) + " |",
             "|" + "|".join("---" for _ in headings) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(k, row[k]) for k, _ in COLUMNS) + " |")
    md_path = os.path.join(out_dir, "results_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return csv_path, md_path, "\n".join(lines)




def in_sweep(method, dt, sweep_dts):
    return any(np.isclose(dt, d) for d in sweep_dts[method])


def sweep_points(rows, method, sweep_dts):
    """Stable, finite-error sweep rows for one method, sorted by dt."""
    pts = [r for r in rows if r["method"] == method and r["stable"]
           and in_sweep(method, r["dt"], sweep_dts) and np.isfinite(r["rel_l2"])]
    return sorted(pts, key=lambda r: r["dt"])


def runtime_at_error(pts, tol):
    """Interpolate runtime at a given error (log-log). NaN outside the data range."""
    pts = [p for p in pts if np.isfinite(p["runtime_s"])]
    if len(pts) < 2:
        return np.nan
    pts = sorted(pts, key=lambda p: p["rel_l2"])
    errors = np.log([p["rel_l2"] for p in pts])
    times = np.log([p["runtime_s"] for p in pts])
    if not (errors[0] <= np.log(tol) <= errors[-1]):
        return np.nan
    return float(np.exp(np.interp(np.log(tol), errors, times)))


def print_summary(rows, sweep_dts):
    print("\n=== Stability")
    for method in METHODS:
        mine = [r for r in rows if r["method"] == method]
        stable = sorted(r["dt"] for r in mine if r["stable"])
        unstable = sorted(r["dt"] for r in mine if not r["stable"])
        print(f"{LABELS[method]}: largest stable dt = {max(stable):g}"
              if stable else f"{LABELS[method]}: no stable runs")
        if unstable:
            print(f"    smallest unstable dt = {min(unstable):g}")
        else:
            print("    no unstable dt found in the tested range")

    print("\n=== Accuracy (order of accuracy from the sweep)")
    for method in METHODS:
        pts = sweep_points(rows, method, sweep_dts)
        if len(pts) < 2:
            print(f"{LABELS[method]}: not enough points")
            continue
        dts = [p["dt"] for p in pts]
        errs = [p["rel_l2"] for p in pts]
        print(f"{LABELS[method]}: observed order = {observed_order(dts, errs):.2f} "
              f"(local: {', '.join(f'{s:.2f}' for s in local_orders(dts, errs))})")

    print("\n=== Cost at equal accuracy (log-log interpolation of the timed sweep)")
    pts = {m: sweep_points(rows, m, sweep_dts) for m in METHODS}
    errs = {m: [p["rel_l2"] for p in pts[m]] for m in METHODS}
    if any(len(e) < 2 for e in errs.values()):
        print("Not enough timed points.")
        return
    lo = max(min(e) for e in errs.values())
    hi = min(max(e) for e in errs.values())
    if lo >= hi:
        print("The two methods' error ranges do not overlap, so no equal-accuracy comparison.")
        return
    print(f"{'error':>10s} | {'explicit (s)':>12s} | {'spectral (s)':>12s} | spectral / explicit")
    for tol in np.geomspace(lo, hi, 5)[1:4]:
        te = runtime_at_error(pts["euler"], tol)
        ts = runtime_at_error(pts["spectral"], tol)
        print(f"{tol:10.2e} | {te:12.3f} | {ts:12.3f} | {ts / te:.2f}")




def finish(fig, ax, title, xlabel, ylabel, path):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_stability(rows, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    T = rows[0]["T"]
    for method in METHODS:
        mine = sorted((r for r in rows if r["method"] == method), key=lambda r: r["dt"])
        x = [r["dt"] for r in mine]
        y = [T if r["stable"] else r["blowup_at"] for r in mine]
        ax.plot(x, y, "-", color=COLORS[method], alpha=0.5)
        ax.plot([r["dt"] for r in mine if r["stable"]],
                [T for r in mine if r["stable"]], "o", color=COLORS[method],
                label=f"{LABELS[method]}: stable")
        bad = [r for r in mine if not r["stable"]]
        if bad:
            ax.plot([r["dt"] for r in bad], [r["blowup_at"] for r in bad], "x", ms=9,
                    mew=2, color=COLORS[method], label=f"{LABELS[method]}: blows up")
    ax.axhline(T, color="grey", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(0, T * 1.08)
    return finish(fig, ax, "Stability: simulated time reached before blow-up is detected",
                  "time step dt", f"simulated time (T = {T:g} means stable)",
                  os.path.join(out_dir, "fig1_stability.png"))


def plot_accuracy(rows, out_dir, sweep_dts):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    first_point = None
    for method in METHODS:
        pts = sweep_points(rows, method, sweep_dts)
        label = LABELS[method]
        if len(pts) >= 2:
            label += f" (fitted order {observed_order([p['dt'] for p in pts], [p['rel_l2'] for p in pts]):.2f})"
        if pts:
            ax.loglog([p["dt"] for p in pts], [p["rel_l2"] for p in pts], "o-",
                      color=COLORS[method], label=label)
            if first_point is None:
                first_point = (pts[0]["dt"], pts[0]["rel_l2"])
        extra = [r for r in rows if r["method"] == method and r["stable"]
                 and not in_sweep(method, r["dt"], sweep_dts) and np.isfinite(r["rel_l2"])]
        if extra:
            ax.loglog([r["dt"] for r in extra], [r["rel_l2"] for r in extra], "o",
                      mfc="none", color=COLORS[method], label=f"{LABELS[method]}: extra stable runs")
    if first_point:
        dts = np.array([r["dt"] for r in rows])
        grid = np.geomspace(dts.min(), dts.max(), 20)
        ax.loglog(grid, first_point[1] * grid / first_point[0], "k--", lw=0.8,
                  label="slope 1 (first order)")
    return finish(fig, ax, "Accuracy: error against the same method's reference run",
                  "time step dt", "relative L2 error at T",
                  os.path.join(out_dir, "fig2_accuracy.png"))


def plot_runtime(rows, out_dir, sweep_dts):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in METHODS:
        pts = [p for p in sweep_points(rows, method, sweep_dts) if np.isfinite(p["runtime_s"])]
        if pts:
            ax.loglog([p["dt"] for p in pts], [p["runtime_s"] for p in pts], "o-",
                      color=COLORS[method], label=LABELS[method])
    return finish(fig, ax, "Cost: wall time to reach T against time step",
                  "time step dt", "best warm wall time (s)",
                  os.path.join(out_dir, "fig3_runtime.png"))


def plot_cost_vs_error(rows, out_dir, sweep_dts):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in METHODS:
        pts = [p for p in sweep_points(rows, method, sweep_dts) if np.isfinite(p["runtime_s"])]
        if pts:
            ax.loglog([p["rel_l2"] for p in pts], [p["runtime_s"] for p in pts], "o-",
                      color=COLORS[method], label=LABELS[method])
            for p in pts:
                ax.annotate(f"{p['dt']:g}", (p["rel_l2"], p["runtime_s"]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points", color=COLORS[method])
    return finish(fig, ax, "Cost at equal accuracy (labels are dt)",
                  "relative L2 error at T", "best warm wall time (s)",
                  os.path.join(out_dir, "fig4_cost_vs_error.png"))




def main(results_dir=RESULTS_DIR, out_dir=OUT_DIR, ref_dt=REF_DT, sweep_dts=SWEEP_DTS):
    cases = load_all(results_dir)
    if not cases:
        print(f"No .npz files found in {results_dir}")
        return 1
    rows = build_rows(cases, ref_dt)
    csv_path, md_path, table_text = write_tables(rows, out_dir)
    print(table_text)
    print_summary(rows, sweep_dts)
    paths = [
        plot_stability(rows, out_dir),
        plot_accuracy(rows, out_dir, sweep_dts),
        plot_runtime(rows, out_dir, sweep_dts),
        plot_cost_vs_error(rows, out_dir, sweep_dts),
    ]
    print("\nWritten:")
    for path in [csv_path, md_path] + paths:
        print("  " + path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())