# braggpeak

Reproducible, recursively tested proton **Bragg-peak** simulation and
validation platform. Compares fast candidate models against tabulated NIST
PSTAR data and (optionally) OpenGATE/Geant4 Monte Carlo references.

> **Research software only.** Nothing here is validated for clinical use.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,plot]"        # add ",gate" for the OpenGATE reference
```

## Quick start

```bash
# Simulate a 150 MeV pristine peak in a water phantom, write CSV/NPZ/PNG/JSON.
braggpeak run configs/water_150mev.yaml

# Print Bragg metrics for a saved depth-dose CSV.
braggpeak metrics experiments/output/water_150mev.csv
```

Every run is driven by a YAML config that is the single source of truth for its
physics inputs; the output `.meta.json` echoes all of them (energy, materials,
densities, I-values, seed, voxel size, model, version) so results regenerate
deterministically.

## What works today (v0.1)

- **Materials** with density, mean excitation energy `I`, and elemental
  composition (`Z/A` derived from composition, never hard-coded).
- **Stopping power**: first-principles Bethe-Bloch, the Bortfeld analytic
  range-energy law, and log-log PSTAR-table interpolation. Bethe ranges agree
  with NIST PSTAR water CSDA ranges to < 2% over 60–230 MeV; a single fitted
  scale reduces the residual to < 0.15 mm RMS.
- **Two candidate models**: deterministic CSDA transport and a stochastic **SDE
  Monte Carlo** (energy-loss straggling + nuclear removal), in homogeneous or
  layered slabs and 1-D CT profiles.
- **Geant4/OpenGATE reference**: a working `QGSP_BIC_EMZ` proton depth-dose
  adapter (`monte_carlo_gate.py`) that runs each simulation in an isolated
  subprocess. The SDE matches Geant4 range to **sub-0.1 mm** at 150/200 MeV and
  is **~14× faster**.
- **Three benchmark cases**: homogeneous water, bone/lung heterogeneous slab,
  and a patient-like head phantom — all within the range criteria vs Geant4
  and/or NIST PSTAR.
- **Scoring**: peak depth, R90/R80/R50 distal ranges, distal 80–20 falloff,
  FWHM, peak-to-entrance ratio, normalised RMSE/MAE, and a 1-D gamma index.
- **Calibrate → compare → report → gate** loop with CI regression thresholds;
  a separate Geant4 benchmark (`braggpeak gate-benchmark`).
- **Artifacts**: CSV + NPZ + metadata JSON + PNG per run; saved benchmark
  reports.
- **Tests**: 49 physics-sanity / regression / determinism tests (`pytest`);
  Geant4 tests auto-skip without the `gate` extra. See `docs/STATUS.md` for the
  full metrics-vs-criteria table.

## Units & conventions

Depth cm · energy MeV · mass stopping power MeV cm²/g · density g/cm³ · LET
keV/µm. Depth `z` increases along the beam with the entrance face at `z = 0`.
See `braggpeak/units.py`. Dose is never normalised implicitly.

## Reproduce the benchmarks

```bash
braggpeak benchmark --model sde     # fast analytic water ladder (CI regression gate)
braggpeak gate-benchmark            # SDE vs Geant4 (needs: pip install '.[gate]')
```

## Roadmap

Energy-dependent nuclear model to close the 200 MeV dose-shape gap
(γ 2%/2mm), a Geant4 voxelized-CT reference, an ML dose surrogate
(`ml_surrogate.py` schema is ready), and LET/RBE scoring. See `docs/STATUS.md`
for the current metrics-vs-criteria gap analysis.
