You are implementing Module C (the v0 model + training loop) of BraggTransporter v3.1.

READ FIRST: `braggtransporter/INTERFACES.md`, `braggtransporter/schema.py`,
`braggtransporter/config.py` (both FROZEN — do NOT edit). Follow the tensor contract
exactly. Other agents are concurrently writing the data loader
(`braggtransporter/data/dataset.py`, `make_loaders(cfg)`), the metrics
(`braggtransporter/metrics.py`), and baselines — write your code against the
documented interfaces; do not implement those modules yourself.

ONLY create/edit these files (touch nothing else):
- `braggtransporter/models/braggtransporter_v0.py`
- `braggtransporter/train.py`
- `configs/bt_v0_1d.yaml`
- `tests/test_bt_v0.py`

`BraggTransporterV0(nn.Module)` — the humble v0 from the plan:
  physics-prior features are already in `x` (channels 4..8); lift `x` to d_model,
  add learned positional encoding over depth, concat an "energy/fidelity token"
  from `scalars`, run a small TransformerEncoder (config: d_model=128, n_layers=4,
  n_heads=8, d_ff=256), then a COORDINATE-QUERY decoder: an MLP that takes
  (per-depth latent, normalized depth coord) and outputs the queried quantity —
  so the model can predict dose at arbitrary z, not just grid nodes. Deterministic
  heads: dose (softplus, nonnegative), letd, and a pooled r80. Include an identity
  `CalibrationWrapper` hook attribute for later. Same forward signature as baselines.
  `.param_count()`.

`train.py` — `python -m braggtransporter.train --config configs/bt_v0_1d.yaml`:
  build model by name via a small registry that maps
  {"mlp","fno1d","dota","braggtransporter_v0"} to their classes (import lazily);
  MPS device via `config.get_device`; loss = distal-edge-weighted MSE on dose
  (weight the region within ~2 cm proximal of R80 by TrainConfig.distal_edge_weight)
  + MSE on letd + smooth-L1 on r80; AdamW + cosine LR; grad clip; deterministic
  seeds; per-epoch val; save best checkpoint (`experiments/bt/<model>/best.pt`) and
  a metrics JSON. Must run on CPU too (for tests) via `--device cpu` and a
  `--fast` flag (tiny synthetic data if the real HDF5 is absent, so the test is
  self-contained).

`configs/bt_v0_1d.yaml`: sensible defaults matching TrainConfig/ModelConfig.

`tests/test_bt_v0.py`: forward-shape + nonnegative dose + coordinate-query at
off-grid z returns finite; a 2-step training smoke on synthetic tensors reduces
the loss. Run `.venv/bin/python -m pytest tests/test_bt_v0.py -q` until green.
Do NOT run git. Print a summary + v0 param count + test result.
