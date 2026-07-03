"""Validation harness: compare candidate depth-dose curves to a reference.

A *reference* is either the Bortfeld analytic Bragg curve, a PSTAR-derived
range, or a loaded CSV (OpenGATE/Geant4 or measured beam data). A *candidate*
is any :class:`~braggpeak.transport.DepthDose` from the CSDA or SDE models.

:func:`compare_curves` reports the range and shape metrics the success criteria
require: peak-depth error, R80/R90/R50 distal-range error, normalised RMSE/MAE,
and gamma pass rates at 2%/2mm and 3%/3mm. :func:`run_case` wraps a single
candidate-vs-reference run with runtime and peak-memory measurement. Nothing is
normalised implicitly; all errors are reported in mm and percent with sign
conventions documented per field.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from .analytic_bragg import bortfeld_depth_dose
from .scoring import (
    compute_bragg_metrics,
    normalized_rmse,
    normalized_mae,
    gamma_index_1d,
)
from .transport import DepthDose


@dataclass
class Comparison:
    """Candidate-vs-reference metric bundle. Errors are candidate minus reference."""

    peak_depth_err_mm: float
    r90_err_mm: float
    r80_err_mm: float
    r50_err_mm: float
    distal_falloff_diff_mm: float
    rmse_pct: float
    mae_pct: float
    gamma_2pct_2mm: float
    gamma_3pct_3mm: float
    ref_peak_depth_mm: float
    cand_peak_depth_mm: float
    ref_r80_mm: float
    cand_r80_mm: float

    def as_dict(self) -> dict:
        return asdict(self)


def _resample_to(
    z_target: NDArray[np.float64],
    z_src: NDArray[np.float64],
    dose_src: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Linearly resample ``dose_src`` (on ``z_src``) onto ``z_target``."""
    return np.interp(z_target, z_src, dose_src, left=0.0, right=0.0)


def compare_curves(
    z_ref_cm: NDArray[np.float64],
    dose_ref: NDArray[np.float64],
    z_cand_cm: NDArray[np.float64],
    dose_cand: NDArray[np.float64],
    *,
    low_dose_threshold: float = 0.01,
) -> Comparison:
    """Compute range/shape/gamma metrics of candidate against reference.

    Both curves are peak-normalised for RMSE/MAE/gamma; the candidate is
    resampled onto the reference grid for the pointwise metrics. Distal-range
    errors are computed from each curve's own metrics (grid-independent).
    """
    ref_m = compute_bragg_metrics(z_ref_cm, dose_ref)
    cand_m = compute_bragg_metrics(z_cand_cm, dose_cand)

    dose_cand_on_ref = _resample_to(z_ref_cm, z_cand_cm, dose_cand)
    rmse = normalized_rmse(dose_ref, dose_cand_on_ref, low_dose_threshold=low_dose_threshold)
    mae = normalized_mae(dose_ref, dose_cand_on_ref, low_dose_threshold=low_dose_threshold)
    g22, _ = gamma_index_1d(z_ref_cm, dose_ref, z_cand_cm, dose_cand,
                            dose_tol_pct=2.0, dta_mm=2.0)
    g33, _ = gamma_index_1d(z_ref_cm, dose_ref, z_cand_cm, dose_cand,
                            dose_tol_pct=3.0, dta_mm=3.0)

    return Comparison(
        peak_depth_err_mm=cand_m.peak_depth_mm - ref_m.peak_depth_mm,
        r90_err_mm=cand_m.r90_mm - ref_m.r90_mm,
        r80_err_mm=cand_m.r80_mm - ref_m.r80_mm,
        r50_err_mm=cand_m.r50_mm - ref_m.r50_mm,
        distal_falloff_diff_mm=cand_m.distal_falloff_80_20_mm - ref_m.distal_falloff_80_20_mm,
        rmse_pct=rmse,
        mae_pct=mae,
        gamma_2pct_2mm=g22,
        gamma_3pct_3mm=g33,
        ref_peak_depth_mm=ref_m.peak_depth_mm,
        cand_peak_depth_mm=cand_m.peak_depth_mm,
        ref_r80_mm=ref_m.r80_mm,
        cand_r80_mm=cand_m.r80_mm,
    )


def bortfeld_reference(
    energy_mev: float,
    z_cm: NDArray[np.float64],
    *,
    energy_spread_pct: float = 0.8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (z_cm, dose) for the Bortfeld analytic reference on ``z_cm``."""
    curve = bortfeld_depth_dose(energy_mev, z_cm, energy_spread_pct=energy_spread_pct)
    return curve.z_cm, curve.dose


@dataclass
class CaseResult:
    """One candidate-vs-reference case with performance and provenance."""

    name: str
    comparison: Comparison
    runtime_s: float
    peak_memory_mb: float
    metadata: dict

    def as_dict(self) -> dict:
        d = {
            "name": self.name,
            "runtime_s": self.runtime_s,
            "peak_memory_mb": self.peak_memory_mb,
            "metadata": self.metadata,
        }
        d.update(self.comparison.as_dict())
        return d


def run_case(
    name: str,
    candidate_factory: Callable[[], DepthDose],
    reference_fn: Callable[[NDArray[np.float64]], tuple[NDArray, NDArray]],
    *,
    low_dose_threshold: float = 0.01,
) -> CaseResult:
    """Run a candidate model, compare to a reference, and time/measure it.

    ``candidate_factory`` produces the candidate DepthDose (timed). The
    reference is evaluated on the candidate's depth grid so both share voxels.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    candidate = candidate_factory()
    runtime = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    z_ref, dose_ref = reference_fn(candidate.z_cm)
    comparison = compare_curves(
        z_ref, dose_ref, candidate.z_cm, candidate.dose,
        low_dose_threshold=low_dose_threshold,
    )
    return CaseResult(
        name=name,
        comparison=comparison,
        runtime_s=runtime,
        peak_memory_mb=peak_mem / 1e6,
        metadata=candidate.metadata,
    )
