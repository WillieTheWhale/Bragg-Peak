You are implementing Module B (baseline models) of BraggTransporter v3.1.

READ FIRST: `braggtransporter/INTERFACES.md`, `braggtransporter/schema.py`,
`braggtransporter/config.py` (both FROZEN — do NOT edit). You do not need braggpeak.

ONLY create/edit these files (touch nothing else):
- `braggtransporter/models/mlp.py`
- `braggtransporter/models/fno1d.py`
- `braggtransporter/models/dota_transformer.py`
- `tests/test_bt_models_baseline.py`

Every model is an `nn.Module` with:
  `forward(x: (B,Nz,9), scalars: (B,4)) -> dict{"dose":(B,Nz), "letd":(B,Nz), "r80":(B,)}`
and a `.param_count() -> int`. `x` channels are the 9 per-depth features
(4 material + 5 prior) in the schema order; `scalars` are (energy, spread, spot,
fidelity). Dose head ends in softplus (nonnegative). Keep models small
(target < 5M params each) — this runs on Apple MPS / 48 GB.

- `MLPBaseline`: per-depth shared MLP; broadcast scalars (small MLP embed) to every
  depth; predict per-depth dose/letd; r80 from a pooled head.
- `FNO1d`: a real 1-D Fourier Neural Operator (spectral conv layers with mode
  truncation, e.g. 16 modes, lifting/projection MLPs, 4 layers). This is the
  baseline the project argues is weak at the sharp distal edge — implement it
  faithfully so the comparison is fair. Condition on scalars via a lifted channel.
- `DoTATransformer`: treat the Nz depth positions as a token sequence, add a
  learned positional encoding and one extra token built from `scalars` (the
  "energy token"), a few TransformerEncoder layers (batch_first), then per-token
  heads for dose/letd and a pooled head for r80.

Constraints: pure torch, CPU-testable, deterministic given a seed. Use
`.venv/bin/python`. Write `tests/test_bt_models_baseline.py` checking: forward
shapes for B=2,Nz=64; dose >= 0; param_count > 0; gradient flows (loss.backward
produces finite grads). Run `.venv/bin/python -m pytest tests/test_bt_models_baseline.py -q`
until green. Do NOT run git. Print a short summary + param counts + test result.
