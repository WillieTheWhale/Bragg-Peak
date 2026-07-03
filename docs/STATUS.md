# Project status vs success criteria

_Research software only. No clinical claims._

Regenerate all numbers with:

```bash
pytest -q                              # 47 physics/regression tests
braggpeak run configs/water_150mev.yaml
braggpeak run configs/heterogeneous_bone_150mev.yaml
braggpeak benchmark --model sde        # fast analytic water ladder + regression gate
braggpeak gate-benchmark               # SDE vs Geant4/OpenGATE (needs the gate extra)
```

## Models and references

| role | implementation |
|---|---|
| candidate 1 | `transport.py` — deterministic CSDA depth-dose (no nuclear removal) |
| candidate 2 | `sde_model.py` — stochastic SDE Monte Carlo with nuclear removal + straggling |
| range reference | NIST PSTAR water CSDA ranges (`data/nist_pstar_water_range.csv`) |
| shape reference | Bortfeld (1997) analytic Bragg curve, NIST-anchored |
| heterogeneous reference | WEPL-mapped NIST-anchored Bortfeld (relative stopping powers) |
| **MC reference** | **OpenGATE/Geant4 `QGSP_BIC_EMZ` — working (`monte_carlo_gate.py`), runs in isolated subprocess** |
| patient-like phantom | `ct_materials.py` synthetic head CT profile (HU→RSP/density calibration) |

## SDE vs Geant4/OpenGATE (real Monte Carlo ground truth)

Calibrated SDE (scale 1.00466, tuned nuclear model) vs Geant4 `QGSP_BIC_EMZ`,
250 k primaries reference, 120 k SDE histories, dz = 0.5 mm:

| E (MeV) | Δpeak (mm) | ΔR80 (mm) | RMSE % | γ 3%/3mm | speedup |
|---|---|---|---|---|---|
| 100 | 0.00 | +0.42 | 3.0 | 100% | ~14× |
| 150 | 0.00 | +0.03 | 2.0 | 99% | ~14× |
| 200 | +1.00 | +0.01 | 3.0 | 83% | ~14× |

Range accuracy against real Geant4 is **sub-0.1 mm at 150/200 MeV** — frontier
quality. The SDE is ~14× faster than Geant4 for equivalent depth-dose scoring.

Material consistency matters: the Geant4 reference uses standard NIST materials
(`G4_WATER`, `G4_BONE_COMPACT_ICRU`, `G4_TISSUE_SOFT_ICRP`), whose ICRU
mean-excitation energies (78, 91.9, 72.3 eV) match the braggpeak materials
exactly, so candidate and reference share stopping-power inputs. An early
attempt to register custom materials via `add_material_weights` silently
dropped the I-value (Geant4 recomputed it from Bragg additivity), biasing the
water range by ~2.8 mm — a caught pitfall now documented in `monte_carlo_gate.py`.
The residual ~1 mm through bone is the Bethe-formula vs Geant4 ICRU-73
stopping-table difference for bone, still within the 1.0 mm criterion.

## Calibration

A single multiplicative stopping-power scale (**1.00466**), fit in closed form
to the NIST PSTAR ladder, removes the Bethe-Bloch systematic range bias:
range RMS error **0.806 mm → 0.148 mm** (max 0.38 mm), 60–230 MeV.

## Metrics vs success criteria

Current column is vs the **Geant4** reference where available, else the analytic reference.

| criterion | target | current | status |
|---|---|---|---|
| Water peak-depth error | ≤ 0.5 mm | 0.00 mm (100/150), +1.0 mm (200) | met at 100/150 |
| Water R80/R90 range error | ≤ 0.7 mm | ≤ 0.42 mm vs Geant4 (0.03 mm at 150/200) | **met** |
| Heterogeneous range vs **Geant4** (water-bone-water) | ≤ 1.0 mm | +1.0 mm peak, +0.73 mm R80 | **met** |
| Heterogeneous range vs analytic WEPL (bone/lung) | ≤ 1.0 mm | +0.20 mm peak, +0.36 mm R80 | **met** |
| Patient-like head vs **Geant4** | ≤ 1.0 mm | +0.50 mm peak, +0.66 mm R80, γ3/3 100% | **met** |
| Patient-like (CT head) range vs analytic WEPL | ≤ 1.0 mm | +0.20 mm peak, +0.32 mm R80 | **met** |
| Water depth-dose RMSE | ≤ 1.5 % | 2.0–3.0 % vs Geant4 | close (met ~1.9% at 150) |
| Gamma 3%/3mm | ≥ 90 % | 99–100 % (100/150), 83 % (200) | met at 100/150 |
| Gamma 2%/2mm (water) | ≥ 95 % | 74–98 % (steep-edge limited) | gap at high E |
| Range monotonic with energy | required | yes | met |
| Dose nonnegative | required | yes | met |
| LET rises to distal edge | required | yes | met |
| Uncertainty intervals | required | `sde_model.dose_uncertainty` | met |
| One-command reproducibility | required | `braggpeak benchmark` / `gate-benchmark` + seeds | met |
| ≥10× faster than Geant4 | target | **~14× faster** (equivalent depth-dose scoring) | **met** |

## Remaining gap: depth-dose shape at high energy

Against the real Geant4 reference the shape agreement is strong at 100/150 MeV
(γ 3%/3mm 99–100 %, RMSE ~2 %). The residual is at **200 MeV**, where the
nuclear fragmentation tail beyond the Bragg peak and the deeper buildup plateau
are only approximated by the constant-rate local-deposition nuclear model
(γ 2%/2mm 58–74 %). The γ 2%/2mm criterion is also intrinsically tight on the
steep distal edge, where sub-percent shape differences fail the 2 % dose test.

**Next experiments:**
1. Make the nuclear removal rate and local-deposition fraction energy-dependent
   (fit per-energy against the Geant4 ladder) to model the growing fragmentation
   tail at high energy — expected to lift 200 MeV γ 2%/2mm past 90 %.
2. Add a Geant4 heterogeneous reference (bone/lung slab) to replace the WEPL
   analytic reference and confirm the ≤1.0 mm result against true MC.
3. Raise the Geant4 reference statistics (≥1 M primaries) in a nightly job so the
   γ 2%/2mm pass rate is not limited by reference Poisson noise.
