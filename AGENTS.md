# AGENTS.md — braggpeak project norms

## Mission

Build a reproducible, recursively tested research platform for proton Bragg-peak
simulation. Compare fast candidate models (analytic CSDA, SDE, ML surrogate)
against high-fidelity OpenGATE/Geant4 references and tabulated NIST PSTAR data.
**Research software only — no clinical claims, ever.**

## Non-negotiable rules

- Be explicit about **units, RNG seeds, materials (density + I-value +
  composition), coordinate frames, beam energy, and voxel size**. See
  `braggpeak/units.py` for the canonical unit set. Never silently normalise
  dose or range — normalisation is always an explicit call.
- Keep reference (OpenGATE/Geant4) outputs **separate** from candidate-model
  outputs. Never overwrite one with the other.
- Every experiment writes numeric artifacts (CSV **and** NPZ) plus a metadata
  JSON, so any plot can be regenerated from saved numbers.
- Prefer small, typed, documented Python modules with tests. Pure
  Python + NumPy/SciPy first; isolate optional deps (`opengate`, `torch`,
  `cupy`) behind extras and import guards.
- Run `pytest` before summarising changes. Add a regression/physics test with
  every new physics behaviour.
- Deterministic seeds everywhere; a run must replay bit-for-bit from its config.

## Architecture

```
braggpeak/
  units.py            unit conventions + physical constants
  materials.py        Material dataclass, reference media, Z/A from composition
  stopping_power.py   Bethe-Bloch, Bortfeld range law, PSTAR table interpolation
  transport.py        deterministic CSDA depth-dose (candidate model 1)
  sde_model.py        stochastic SDE transport (candidate model 2)   [planned]
  monte_carlo_gate.py OpenGATE adapter (reference)                    [planned]
  ml_surrogate.py     optional torch surrogate                        [planned]
  scoring.py          R80/R90/R50, peak depth, FWHM, RMSE/MAE, gamma
  validation.py       run/compare/report loop + regression gates      [planned]
  io.py               CSV/NPZ/PNG/JSON artifact writers
  cli.py              `braggpeak run|metrics|version`
configs/   YAML run configs (single source of truth per run)
experiments/  generated outputs (gitignored)
tests/     physics-sanity + regression tests
docs/      notes and reports
```

## Reproducibility workflow

1. One YAML config per run under `configs/`.
2. `braggpeak run configs/<case>.yaml` → CSV + NPZ + PNG + meta JSON.
3. Metadata records every input: energy, materials, densities, I-values, seed,
   dz, model name, package version.
4. Regression tests compare metrics against saved thresholds and fail on drift.

## Success criteria (for claiming a new best model)

- Water peak-depth error ≤ 0.5 mm; R80/R90 distal range error ≤ 0.7 mm.
- Heterogeneous distal range & peak-depth error ≤ 1.0 mm vs OpenGATE/Geant4.
- Depth-dose RMSE ≤ 1.5% (water), ≤ 2.5% (heterogeneous), excluding documented
  low-dose tails.
- Gamma ≥ 95% at 2%/2mm (water), ≥ 90% at 3%/3mm (heterogeneous).
- Range monotonic with energy; dose nonnegative; LET rises to the distal edge;
  uncertainty intervals reported.
- ≥ 10× faster than the chosen reference for equivalent scoring, or document why
  accuracy was prioritised.
- All benchmarks regenerate from one CLI command with pinned deps + seeds.

## Commit conventions

- Small, frequent commits with concise messages (`add water phantom scorer`,
  `validate pstar interpolation`, `cap final-step deposition`).
- No attribution to tools or assistants in commits, PRs, or file headers.
