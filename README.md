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
  with NIST PSTAR water CSDA ranges to < 2% over 60–230 MeV.
- **Transport**: deterministic CSDA pencil-beam depth-dose with range
  straggling and beam energy spread, in homogeneous or layered slabs.
- **Scoring**: peak depth, R90/R80/R50 distal ranges, distal 80–20 falloff,
  FWHM, peak-to-entrance ratio, normalised RMSE/MAE, and a 1-D gamma index.
- **Artifacts**: CSV + NPZ + metadata JSON + PNG for every run.
- **Tests**: 24 physics-sanity / regression / determinism tests (`pytest`).

## Units & conventions

Depth cm · energy MeV · mass stopping power MeV cm²/g · density g/cm³ · LET
keV/µm. Depth `z` increases along the beam with the entrance face at `z = 0`.
See `braggpeak/units.py`. Dose is never normalised implicitly.

## Roadmap

SDE stochastic transport, OpenGATE reference adapter, heterogeneous-slab and
patient-like benchmarks, the iterative calibrate→compare→report loop, and CI
regression gates. See `AGENTS.md`.
