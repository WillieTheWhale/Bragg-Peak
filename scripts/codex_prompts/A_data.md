You are implementing Module A (data engine) of BraggTransporter v3.1.

READ FIRST: `braggtransporter/INTERFACES.md`, `braggtransporter/schema.py`,
`braggtransporter/config.py` (schema.py and config.py are FROZEN — do NOT edit them),
and the sibling physics package `braggpeak/` (especially `sde_model.py`,
`transport.py`, `stopping_power.py`, `analytic_bragg.py`, `scoring.py`,
`materials.py`, `units.py`). Reuse braggpeak physics — do not reimplement it.

ONLY create/edit these files (touch nothing else):
- `braggtransporter/physics_prior.py`
- `braggtransporter/data/physics_engine.py`
- `braggtransporter/data/generate.py`
- `braggtransporter/data/dataset.py`
- `tests/test_bt_data.py`

Implement exactly the signatures in INTERFACES.md "Module A". Details:
- `physics_prior.compute_prior(z_cm, material_profile, beam)` returns the 5
  `PRIOR_FIELDS` arrays (wepl_cm, csda_stopping, bortfeld_dose, depth_over_r0,
  resid_energy_mev) computed from braggpeak analytic/CSDA functions. bortfeld_dose
  normalised so its peak == 1.
- `simulate_sample(...)` builds a `schema.Sample` for a water or layered-slab
  geometry: material profile (density/rsp/i_value/material_class per depth), the
  prior, an SDE (fidelity=SDE) or analytic (fidelity=ANALYTIC) dose target, a LETd
  target (use braggpeak LET if available, else a physically monotone-to-distal
  proxy and note it), and scalar R80 via braggpeak.scoring. Call sample.validate().
- `generate.py` CLI writes an HDF5 with groups train/val/heldout_energy using
  DataConfig defaults (17 train energies, 4 held-out). Geometries = water plus
  randomized bone/lung/air slabs (seeded). Store everything needed to reconstruct
  Samples. Print a summary (counts, shapes).
- `dataset.py: BraggDataset` yields the exact tensor dict in INTERFACES.md;
  `make_loaders(cfg)` returns (train,val,heldout) DataLoaders. Standardize input
  channels with train-split statistics saved into the HDF5/へ a norm json.

Constraints: deterministic seeds; nonnegative dose; explicit units; float32 tensors
out of the Dataset, float64 in the Sample. Use `.venv/bin/python` to test.
Run `.venv/bin/python -m pytest tests/test_bt_data.py -q` and
`.venv/bin/python -m braggtransporter.data.generate` (small, e.g. 4 energies × 8
geometries for the smoke run) until both pass. Do NOT run git. Keep it fast (CPU).
When done, print a one-paragraph summary of what you built and the test result.
