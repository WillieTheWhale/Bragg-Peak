"""Iterative benchmark loop: baseline -> calibrate -> compare -> report -> gate.

Runs the water-phantom energy ladder for a candidate model, first uncalibrated
then with a fitted stopping-power scale, comparing each against the
NIST-anchored Bortfeld reference. Writes a Markdown report plus a machine
metrics JSON, and evaluates the run against the project success thresholds so
CI can fail on regression.

One command regenerates everything: ``braggpeak benchmark``.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__
from .calibrate import fit_stopping_scale, calibrated_stopping_factory, nist_range_cm
from .materials import WATER
from .sde_model import simulate_depth_dose_sde
from .transport import Slab, simulate_depth_dose
from .validation import nist_anchored_reference, run_case, compare_curves


# Success-criteria thresholds for the homogeneous water phantom.
WATER_THRESHOLDS = {
    "peak_depth_err_mm": 0.5,
    "r80_err_mm": 0.7,
    "r90_err_mm": 0.7,
    "rmse_pct": 1.5,
    "gamma_2pct_2mm": 0.95,  # minimum pass fraction
}

DEFAULT_LADDER = [80.0, 100.0, 150.0, 200.0, 230.0]


def _candidate_factory(model: str, energy: float, scale: float, dz: float) -> Callable:
    depth = nist_range_cm(energy) * 1.25
    factory = calibrated_stopping_factory(scale)
    if model == "sde":
        return lambda: simulate_depth_dose_sde(
            energy, [Slab(WATER, depth)], dz_cm=dz, n_histories=40000,
            seed=1234, stopping_model_factory=factory,
        )
    if model == "csda":
        return lambda: simulate_depth_dose(
            energy, [Slab(WATER, depth)], dz_cm=dz,
            stopping_model_factory=factory,
        )
    raise ValueError(f"Unknown model '{model}'.")


def run_water_ladder(
    model: str = "sde",
    energies: list[float] | None = None,
    scale: float = 1.0,
    dz_cm: float = 0.02,
    label: str = "baseline",
) -> list[dict]:
    """Run the ladder at a fixed calibration scale; return per-energy records."""
    energies = energies or DEFAULT_LADDER
    records = []
    for e in energies:
        ref_fn = lambda z, _e=e: nist_anchored_reference(_e, z)
        case = run_case(
            f"{label}_{model}_{int(e)}MeV",
            _candidate_factory(model, e, scale, dz_cm),
            ref_fn,
        )
        rec = case.as_dict()
        rec["energy_mev"] = e
        records.append(rec)
    return records


def evaluate_thresholds(records: list[dict]) -> tuple[bool, list[str]]:
    """Check per-energy records against WATER_THRESHOLDS. Returns (passed, msgs)."""
    msgs = []
    passed = True
    for r in records:
        e = r["energy_mev"]
        for key, limit in WATER_THRESHOLDS.items():
            val = r[key]
            if key == "gamma_2pct_2mm":
                ok = val >= limit
                cmp = ">="
            else:
                ok = abs(val) <= limit
                cmp = "<="
            if not ok:
                passed = False
                msgs.append(f"{e:.0f} MeV: {key}={val:.3f} violates {cmp}{limit}")
    return passed, msgs


# Allowed worsening before a metric counts as a regression (CI gate tolerance).
REGRESSION_TOL = {
    "peak_depth_err_mm": 0.15,
    "r80_err_mm": 0.15,
    "r90_err_mm": 0.15,
    "rmse_pct": 0.5,
    "gamma_2pct_2mm": 0.05,  # pass fraction may not drop by more than this
}


def check_regression(
    calibrated: list[dict],
    baseline_path: str | Path,
) -> tuple[bool, list[str]]:
    """Fail if any per-energy metric degraded beyond REGRESSION_TOL vs baseline.

    If no committed baseline exists yet, this returns (True, []) so the first
    run can establish one. Compares by energy so re-ordering is safe.
    """
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        return True, ["no committed baseline; establishing one"]
    prev = {r["energy_mev"]: r for r in json.loads(baseline_path.read_text())["calibrated"]}
    msgs = []
    ok = True
    for r in calibrated:
        e = r["energy_mev"]
        if e not in prev:
            continue
        for key, tol in REGRESSION_TOL.items():
            if key == "gamma_2pct_2mm":
                if r[key] < prev[e][key] - tol:
                    ok = False
                    msgs.append(f"{e:.0f} MeV: {key} {prev[e][key]:.3f}->{r[key]:.3f} regressed")
            else:
                if abs(r[key]) > abs(prev[e][key]) + tol:
                    ok = False
                    msgs.append(f"{e:.0f} MeV: {key} {prev[e][key]:.3f}->{r[key]:.3f} regressed")
    return ok, msgs


@dataclass
class BenchmarkReport:
    scale: float
    baseline: list[dict]
    calibrated: list[dict]
    passed: bool
    violations: list[str]
    provenance: dict
    regression_ok: bool = True
    regression_msgs: list[str] | None = None


def run_benchmark(
    out_dir: str | Path = "benchmarks/water_ladder",
    model: str = "sde",
    energies: list[float] | None = None,
    dz_cm: float = 0.02,
) -> BenchmarkReport:
    """Full loop: baseline ladder, fit scale, calibrated ladder, report, gate."""
    energies = energies or DEFAULT_LADDER
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    baseline = run_water_ladder(model, energies, scale=1.0, dz_cm=dz_cm, label="baseline")
    cal = fit_stopping_scale()
    calibrated = run_water_ladder(model, energies, scale=cal.stopping_scale, dz_cm=dz_cm,
                                  label="calibrated")
    passed, violations = evaluate_thresholds(calibrated)
    baseline_metrics_path = out / "baseline_metrics.json"
    regression_ok, regression_msgs = check_regression(calibrated, baseline_metrics_path)
    wall = time.perf_counter() - t0

    provenance = {
        "package_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "model": model,
        "energies_mev": energies,
        "dz_cm": dz_cm,
        "stopping_scale": cal.stopping_scale,
        "calibration": cal.as_dict(),
        "reference": "NIST-anchored Bortfeld analytic Bragg curve",
        "thresholds": WATER_THRESHOLDS,
        "wall_time_s": wall,
    }
    report = BenchmarkReport(
        cal.stopping_scale, baseline, calibrated, passed, violations, provenance,
        regression_ok=regression_ok, regression_msgs=regression_msgs,
    )

    metrics_payload = {
        "provenance": provenance,
        "baseline": baseline,
        "calibrated": calibrated,
        "passed": passed,
        "violations": violations,
        "regression_ok": regression_ok,
        "regression_msgs": regression_msgs,
    }
    (out / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, default=float))
    # Establish the committed regression baseline on first run.
    if not baseline_metrics_path.exists():
        baseline_metrics_path.write_text(json.dumps(metrics_payload, indent=2, default=float))
    (out / "report.md").write_text(_render_markdown(report))
    return report


def run_gate_benchmark(
    out_dir: str | Path = "benchmarks/gate_water",
    energies: list[float] | None = None,
    dz_cm: float = 0.05,
    n_primaries: int = 200000,
    n_histories: int = 120000,
) -> dict:
    """Compare the calibrated SDE model against a Geant4/OpenGATE reference.

    For each energy, generates (or reuses) a cached high-statistics Geant4
    depth-dose, runs the calibrated SDE, and reports range/shape/gamma metrics
    plus the SDE-vs-Geant4 runtime speedup. Heavy (needs a Geant4 backend), so
    this is a nightly/manual benchmark, separate from the fast CI loop.
    """
    from .monte_carlo_gate import opengate_available, simulate_depth_dose_gate

    if not opengate_available():
        raise RuntimeError("Geant4 benchmark requires the opengate extra.")
    energies = energies or [100.0, 150.0, 200.0]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref_dir = out / "gate_refs"
    ref_dir.mkdir(exist_ok=True)

    cal = fit_stopping_scale()
    factory = calibrated_stopping_factory(cal.stopping_scale)
    records = []
    for e in energies:
        depth = nist_range_cm(e) * 1.35
        ref_path = ref_dir / f"water_{int(e)}.npz"
        t_gate = None
        if ref_path.exists():
            data = np.load(ref_path)
            z_ref, dose_ref = data["z_cm"], data["dose"]
        else:
            t0 = time.perf_counter()
            g = simulate_depth_dose_gate(e, [Slab(WATER, depth)], dz_cm=dz_cm,
                                         n_primaries=n_primaries, seed=1)
            t_gate = time.perf_counter() - t0
            z_ref, dose_ref = g.z_cm, g.dose
            np.savez(ref_path, z_cm=z_ref, dose=dose_ref, runtime_s=t_gate)

        t1 = time.perf_counter()
        s = simulate_depth_dose_sde(e, [Slab(WATER, depth)], dz_cm=dz_cm,
                                    n_histories=n_histories, seed=1234,
                                    stopping_model_factory=factory)
        t_sde = time.perf_counter() - t1
        comp = compare_curves(z_ref, dose_ref, s.z_cm, s.dose, low_dose_threshold=0.01)
        rec = comp.as_dict()
        rec["energy_mev"] = e
        rec["sde_runtime_s"] = t_sde
        rec["gate_runtime_s"] = t_gate
        rec["speedup"] = (t_gate / t_sde) if t_gate else None
        records.append(rec)

    payload = {
        "reference": "OpenGATE/Geant4 QGSP_BIC_EMZ",
        "stopping_scale": cal.stopping_scale,
        "package_version": __version__,
        "n_primaries": n_primaries,
        "n_histories": n_histories,
        "records": records,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=float))
    (out / "report.md").write_text(_render_gate_markdown(payload))
    return payload


def _render_gate_markdown(payload: dict) -> str:
    lines = [
        "# Water-phantom benchmark vs Geant4/OpenGATE",
        "",
        "> Research software only. No clinical claims.",
        "",
        f"- Reference: {payload['reference']}  ·  "
        f"{payload['n_primaries']} primaries",
        f"- Candidate: calibrated SDE (scale {payload['stopping_scale']:.5f}), "
        f"{payload['n_histories']} histories",
        "",
        "| E (MeV) | Δpeak (mm) | ΔR80 (mm) | ΔR90 (mm) | RMSE % | γ 2/2 (%) | "
        "γ 3/3 (%) | SDE t (s) | speedup |",
        "|" + "---|" * 9,
    ]
    for r in payload["records"]:
        sp = f"{r['speedup']:.1f}x" if r.get("speedup") else "cached"
        lines.append(
            f"| {r['energy_mev']:.0f} | {r['peak_depth_err_mm']:+.2f} | "
            f"{r['r80_err_mm']:+.2f} | {r['r90_err_mm']:+.2f} | {r['rmse_pct']:.2f} | "
            f"{r['gamma_2pct_2mm']*100:.0f} | {r['gamma_3pct_3mm']*100:.0f} | "
            f"{r['sde_runtime_s']:.2f} | {sp} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt_row(r: dict) -> str:
    return (
        f"| {r['energy_mev']:.0f} | {r['cand_peak_depth_mm']:.2f} | "
        f"{r['ref_peak_depth_mm']:.2f} | {r['peak_depth_err_mm']:+.2f} | "
        f"{r['r80_err_mm']:+.2f} | {r['r90_err_mm']:+.2f} | {r['rmse_pct']:.2f} | "
        f"{r['gamma_2pct_2mm']*100:.1f} | {r['runtime_s']*1000:.0f} |"
    )


def _render_markdown(report: BenchmarkReport) -> str:
    p = report.provenance
    lines = [
        "# Water-phantom Bragg-peak benchmark",
        "",
        "> Research software only. No clinical claims.",
        "",
        f"- Package version: `{p['package_version']}`  ·  model: `{p['model']}`  ·  "
        f"numpy `{p['numpy']}`, python `{p['python']}`",
        f"- Reference: {p['reference']}",
        f"- Fitted stopping-power scale: **{report.scale:.5f}**",
        f"- Calibration baseline range RMS: "
        f"{p['calibration']['baseline_rms_mm']:.3f} mm -> calibrated "
        f"{p['calibration']['calibrated_rms_mm']:.3f} mm",
        f"- Wall time: {p['wall_time_s']:.1f} s",
        "",
        f"## Result: {'PASS' if report.passed else 'FAIL'}",
        "",
        "> Shape metrics (RMSE, gamma) here are against the **Bortfeld analytic**"
        " reference, whose plateau-to-peak ratio is conservative; the"
        " `gate-benchmark` (vs Geant4) is authoritative for dose shape. Range"
        " metrics below are reliable against either reference.",
        "",
    ]
    if report.violations:
        lines.append("Threshold violations:")
        lines += [f"- {v}" for v in report.violations]
        lines.append("")
    header = ("| E (MeV) | peak (mm) | ref peak | Δpeak (mm) | ΔR80 (mm) | "
              "ΔR90 (mm) | RMSE % | γ 2/2 (%) | t (ms) |")
    sep = "|" + "---|" * 9
    lines += ["## Calibrated ladder vs NIST-anchored reference", "", header, sep]
    lines += [_fmt_row(r) for r in report.calibrated]
    lines += ["", "## Baseline (uncalibrated) ladder", "", header, sep]
    lines += [_fmt_row(r) for r in report.baseline]
    lines += [
        "",
        "## Success thresholds (water)",
        "",
        "| metric | limit |",
        "|---|---|",
    ] + [f"| {k} | {v} |" for k, v in report.provenance["thresholds"].items()]
    lines.append("")
    return "\n".join(lines)
