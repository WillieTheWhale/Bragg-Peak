# Project status vs success criteria

_Research software only. No clinical claims._

Regenerate all numbers with:

```bash
pytest -q                              # 42 physics/regression tests
braggpeak run configs/water_150mev.yaml
braggpeak run configs/heterogeneous_bone_150mev.yaml
braggpeak benchmark --model sde        # water ladder + calibration + gate
```

## Models and references

| role | implementation |
|---|---|
| candidate 1 | `transport.py` — deterministic CSDA depth-dose (no nuclear removal) |
| candidate 2 | `sde_model.py` — stochastic SDE Monte Carlo with nuclear removal + straggling |
| range reference | NIST PSTAR water CSDA ranges (`data/nist_pstar_water_range.csv`) |
| shape reference | Bortfeld (1997) analytic Bragg curve, NIST-anchored |
| heterogeneous reference | WEPL-mapped NIST-anchored Bortfeld (relative stopping powers) |
| MC reference | OpenGATE/Geant4 adapter — planned plug-in (`monte_carlo_gate.py`) |

## Calibration

A single multiplicative stopping-power scale (**1.00466**), fit in closed form
to the NIST PSTAR ladder, removes the Bethe-Bloch systematic range bias:
range RMS error **0.806 mm → 0.148 mm** (max 0.38 mm), 60–230 MeV.

## Metrics vs success criteria

| criterion | target | current | status |
|---|---|---|---|
| Water peak-depth error | ≤ 0.5 mm | ~0.2–0.8 mm (0.8 at 150) | mostly met |
| Water R80/R90 range error | ≤ 0.7 mm | < 0.2 mm (80–200 MeV); ~1.0 at 230 | met except 230 MeV edge |
| Heterogeneous range error (bone/lung) | ≤ 1.0 mm | +0.20 mm peak, +0.36 mm R80 | **met** |
| Water depth-dose RMSE | ≤ 1.5 % | ~4–5 % | **gap** |
| Gamma 2%/2mm (water) | ≥ 95 % | ~20–70 % | **gap** |
| Range monotonic with energy | required | yes | met |
| Dose nonnegative | required | yes | met |
| LET rises to distal edge | required | yes | met |
| Uncertainty intervals | required | `sde_model.dose_uncertainty` | met |
| One-command reproducibility | required | `braggpeak benchmark` + pinned seeds | met |
| ≥10× faster than Geant4 | target | SDE ladder ~8 s CPU; Geant4 not yet wired | pending reference |

## Known gap: depth-dose shape

The SDE Bragg peak is narrower (FWHM ~17.5 vs 19.2 mm at 150 MeV) with a steeper
proximal shoulder than the Bortfeld reference, leaving a ~3–5 % normalized RMSE
concentrated in the plateau and shoulder. Range accuracy (the clinically hardest
metric) is met; the shape gap reflects the simplified nuclear-secondary and
straggling model.

**Next experiment:** wire the OpenGATE/Geant4 reference (`monte_carlo_gate.py`),
then calibrate the nuclear removal rate, local-deposition fraction, and
straggling width against Geant4 depth-dose rather than the Bortfeld analytic
form. This is expected to close the plateau-shape RMSE and raise the gamma pass
rate toward the 95 % target.
