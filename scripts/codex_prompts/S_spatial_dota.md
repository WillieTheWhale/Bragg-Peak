Build a SPATIALLY-PRESERVING 3-D dose model to replace the lateral-blind DoTA3D.

ROOT CAUSE (see docs/PLATEAU_DIAGNOSIS.md): current braggtransporter/models/dota3d.py
encodes each transverse slice with AdaptiveAvgPool2d(1) -> the model is provably blind to
lateral dose position (encoder token diff 6e-8 for a laterally-shifted blob), so it can
only emit a blurry lateral blob and gamma plateaus ~63%. Fix: preserve (H,W) structure.

READ FIRST: braggtransporter/models/dota3d.py (forward contract to MATCH exactly:
forward(x:(B,C,D,H,W), scalars:(B,4)) -> {"dose":(B,D,H,W)}, softplus output, .param_count()),
braggtransporter/data/doserad.py (DOSERAD_INPUT_CHANNELS, BEV layout), scripts/
train_doserad_gpu.py (model registry / --model arg).

CREATE braggtransporter/models/dota3d_spatial.py -- `DoTA3DSpatial(nn.Module)`, SAME
forward contract. Design (DoTA-faithful, spatially preserving):
- Per-slice patch embedding: Conv2d(C, d_model, kernel=patch, stride=patch) so each slice
  becomes a grid of P*P patch tokens (NOT global-pooled). Keep the (Hp,Wp) grid.
- Tokens per slice = Hp*Wp; add 2-D lateral positional encoding + a depth positional
  encoding. Prepend the scalar conditioning token (energy/layer/sin,cos gantry).
- A Transformer encoder that mixes across BOTH depth and the patch grid (flatten
  depth*Hp*Wp tokens; if that's too many, use factorized attention: attention along depth
  per patch-position, then a light spatial mix — keep it under ~8M params and MPS+CUDA safe,
  NO grid_sample, NO use_deterministic_algorithms).
- Decoder: reshape tokens back to (B, d_model, D, Hp, Wp) feature maps and upsample with
  conv layers (pixel-shuffle or ConvTranspose per slice) back to (H,W). Softplus output.
- Configurable d_model, n_layers, n_heads, patch_size (default 4). .param_count().

WIRE: add "dota3d_spatial" to the --model registry in scripts/train_doserad_gpu.py
(keep dota3d + bragg3d for back-compat).

CRITICAL TEST (tests/test_bt_dota3d_spatial.py, CPU-fast):
- forward shape (B=2,C=3,D=16,H=24,W=24) -> dose (2,16,24,24), nonnegative, finite backward grads.
- LATERAL SENSITIVITY (the whole point): two inputs identical except a blob at (5,5) vs
  (18,18) must produce MEANINGFULLY DIFFERENT outputs at those locations
  (assert output diff at the two positions >> 1e-3 -- i.e. the model is NOT lateral-blind).
- param_count < 8M; 2-step loss decrease on a synthetic batch; MPS backward must not raise.

VERIFY: `.venv/bin/python -m pytest tests/test_bt_dota3d_spatial.py -q` and a 2-epoch
`--model dota3d_spatial --fast --device cpu` smoke. Do NOT run git. Print param count, the
lateral-sensitivity numbers (must be large), and test results. The orchestrator runs GPU training.
