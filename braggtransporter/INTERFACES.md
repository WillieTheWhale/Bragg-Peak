# BraggTransporter — Fixed Interface Contract (v3.1, Phase 1)

Every module below is implemented against these signatures. `schema.py` and
`config.py` are already written and are FROZEN — do not edit them. Reuse the
existing validated physics in the sibling `braggpeak/` package wherever possible
(`braggpeak.sde_model`, `braggpeak.transport`, `braggpeak.stopping_power`,
`braggpeak.analytic_bragg`, `braggpeak.scoring`, `braggpeak.materials`).

Import root is the repo directory; run everything with `.venv/bin/python`.

## Data contract (schema.py — FROZEN)
- `Sample`, `Fidelity`, `MATERIAL_FIELDS`, `BEAM_FIELDS`, `PRIOR_FIELDS`,
  `QUANTITIES`, `pack_perdepth(sample)->(Nz,9)`, `pack_scalars(sample)->(4,)`,
  `C_IN_PERDEPTH=9`, `C_SCALAR=4`.

## Module A — data engine (`braggtransporter/physics_prior.py`, `data/physics_engine.py`, `data/generate.py`, `data/dataset.py`)
- `physics_prior.compute_prior(z_cm, material_profile, beam) -> dict[str,ndarray]`
  returns the `PRIOR_FIELDS` arrays using braggpeak analytic/CSDA physics.
- `data/physics_engine.py: simulate_sample(energy_mev, geometry, dz_cm, fidelity, seed) -> Sample`
  builds a `Sample` (material profile + prior + SDE/analytic dose + LETd target +
  R80) by calling braggpeak. `geometry` describes water/slab layers.
- `data/generate.py`: CLI `python -m braggtransporter.data.generate --config <yaml|defaults>`
  writes an HDF5 dataset (train/val/heldout-energy splits) to `DataConfig.out_path`.
- `data/dataset.py: BraggDataset(torch.utils.data.Dataset)` reads the HDF5 and
  yields dict tensors: `{"x":(Nz,9),"scalars":(4,),"z":(Nz,),"dose":(Nz,),
  "letd":(Nz,),"r80":(),"fidelity":()}`. Provide `make_loaders(cfg)->(train,val,heldout)`.

## Module B — baselines (`braggtransporter/models/{mlp,fno1d,dota_transformer}.py`)
Every model: `forward(x:(B,Nz,9), scalars:(B,4)) -> dict{"dose":(B,Nz), optional "letd":(B,Nz), optional "r80":(B,)}`.
- `mlp.MLPBaseline` — per-depth MLP + global pooling for scalars.
- `fno1d.FNO1d` — 1-D Fourier Neural Operator (the baseline v2 argues is weak at
  the distal edge; keep it a first-class baseline).
- `dota_transformer.DoTATransformer` — depth-sequence self-attention (DoTA-style),
  scalars as an energy token.
All expose `.param_count()`.

## Module C — v0 model + training (`braggtransporter/models/braggtransporter_v0.py`, `braggtransporter/train.py`, `configs/bt_v0_1d.yaml`)
- `braggtransporter_v0.BraggTransporterV0` — physics-prior features (already in x) +
  simple transformer encoder over depth tokens + coordinate-query decoder +
  deterministic `dose`/`letd`/`r80` heads + softplus nonnegativity on dose. Same
  forward signature as baselines. Include a `calibration wrapper` hook (identity for now).
- `train.py`: `python -m braggtransporter.train --config configs/bt_v0_1d.yaml`.
  MPS device, distal-edge-weighted MSE + R80 loss, AdamW, cosine schedule, grad
  clip, deterministic seeds, checkpoint + metrics JSON to `TrainConfig.out_dir`.

## Module D — metrics/eval/calibration (`braggtransporter/metrics.py`, `calibration.py`, `evaluate.py`)
- `metrics.py`: reuse `braggpeak.scoring` where possible. Provide
  `gamma_index_1d(pred,ref,z,dose_pct,dta_mm)`, `r80_r90_r50(z,dose)`,
  `peak_depth(z,dose)`, `distal_80_20_mm(z,dose)`, `rmse_pct`, and a
  `distal_edge_error_mm(pred,ref,z)` (the headline metric for the v2 claim).
- `calibration.py`: `CalibrationWrapper` (temperature/quantile) + `coverage(...)`.
- `evaluate.py`: `python -m braggtransporter.evaluate --ckpt ... --data ...` prints
  a metrics table (per-energy + held-out) and writes CSV/JSON.

## Tests (`tests/test_bt_*.py`)
Fast CPU-only pytest for: schema packing, prior shapes, dataset roundtrip, each
model forward-shape + nonnegative dose, metrics vs a known analytic curve.

## Non-negotiables (from AGENTS.md)
Explicit units/seeds/materials; deterministic replay; nonnegative dose; every run
writes CSV+NPZ+JSON; add a regression test with every new behaviour.
