"""Tests for the validation harness and benchmark regression gate."""

from __future__ import annotations

import numpy as np

from braggpeak.materials import WATER
from braggpeak.transport import Slab, simulate_depth_dose
from braggpeak.validation import (
    compare_curves,
    bortfeld_reference,
    nist_anchored_reference,
    run_case,
)
from braggpeak.benchmark import check_regression, evaluate_thresholds, WATER_THRESHOLDS


def test_compare_identical_curves():
    z = np.arange(0.05, 20.0, 0.02)
    _, dose = bortfeld_reference(150.0, z)
    c = compare_curves(z, dose, z, dose)
    assert abs(c.peak_depth_err_mm) < 1e-6
    assert c.rmse_pct < 1e-6
    assert c.gamma_2pct_2mm == 1.0


def test_nist_anchored_reference_range_matches_nist():
    from braggpeak.calibrate import nist_range_cm

    z = np.arange(0.05, 20.0, 0.01)
    _, dose = nist_anchored_reference(150.0, z)
    peak_cm = z[int(np.argmax(dose))]
    # Peak sits just proximal to the NIST range.
    assert abs(peak_cm - nist_range_cm(150.0)) < 0.3


def test_run_case_reports_runtime_and_memory():
    case = run_case(
        "csda_150",
        lambda: simulate_depth_dose(150.0, [Slab(WATER, 20.0)], dz_cm=0.05),
        lambda z: nist_anchored_reference(150.0, z),
    )
    assert case.runtime_s > 0
    assert case.peak_memory_mb > 0
    assert "model" in case.metadata


def test_evaluate_thresholds_flags_bad_metrics():
    good = [{
        "energy_mev": 150.0, "peak_depth_err_mm": 0.1, "r80_err_mm": 0.1,
        "r90_err_mm": 0.1, "rmse_pct": 1.0, "gamma_2pct_2mm": 0.99,
    }]
    passed, msgs = evaluate_thresholds(good)
    assert passed and not msgs

    bad = [dict(good[0], r80_err_mm=5.0)]
    passed, msgs = evaluate_thresholds(bad)
    assert not passed and msgs


def test_regression_gate_detects_degradation(tmp_path):
    baseline = {
        "calibrated": [{
            "energy_mev": 150.0, "peak_depth_err_mm": 0.1, "r80_err_mm": 0.1,
            "r90_err_mm": 0.1, "rmse_pct": 1.0, "gamma_2pct_2mm": 0.99,
        }]
    }
    import json
    p = tmp_path / "baseline_metrics.json"
    p.write_text(json.dumps(baseline))

    # A clearly worse run must fail the gate.
    worse = [dict(baseline["calibrated"][0], rmse_pct=5.0)]
    ok, msgs = check_regression(worse, p)
    assert not ok and msgs

    # An equal run passes.
    same = [dict(baseline["calibrated"][0])]
    ok, _ = check_regression(same, p)
    assert ok


def test_regression_gate_no_baseline_passes(tmp_path):
    ok, msgs = check_regression([], tmp_path / "missing.json")
    assert ok
