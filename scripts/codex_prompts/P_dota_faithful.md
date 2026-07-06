Implement a DoTA-faithful beam's-eye-view depth-transformer for DoseRAD2026 3-D
beamlets (overnight goal: close the gap to DoTA/ADoTA gamma). Our current Bragg3D is a
rough approximation; DoTA's PUBLISHED architecture is proven to reach ~99% gamma, so a
faithful reimplementation is the highest-leverage change.

READ FIRST: brag_deep_learning/bragg_peak_frontier_research_pack/papers/
2022_dota_transformer_dose_prediction.md and 2026_adota_angle_dependent_dose_transformer.md
(the architecture), braggtransporter/data/doserad.py (BEV tensor layout
x:(B,C,D,H,W), dose:(B,D,H,W), per-beamlet max-normalized), braggtransporter/models/
bragg3d.py (current model + forward contract), scripts/train_doserad_gpu.py (trainer
that will import the model), braggtransporter/metrics.py, config.py (FROZEN).

DoTA's formulation (implement faithfully in PyTorch, MPS+CUDA safe, NO grid_sample, NO
torch.use_deterministic_algorithms):
- Treat the beamlet as a SEQUENCE of 2D transverse slices along depth D. A shared 2D
  CNN encoder maps each (C,H,W) slice -> a token vector; add a learned depth positional
  encoding; prepend/concat a conditioning token built from beam scalars (energy layer,
  gantry angle). A Transformer encoder routes information ALONG DEPTH between slice
  tokens. A 2D CNN decoder maps each output token back to a (H,W) dose slice; softplus
  (nonnegative). This is the DoTA "CNN-encoder + depth-transformer + CNN-decoder" design.

ONLY create/edit:
- braggtransporter/models/dota3d.py -- `DoTA3D(nn.Module)` with the SAME forward
  contract the trainer expects (match Bragg3D: forward(x, scalars)->{"dose":(B,D,H,W)}
  or the exact signature bragg3d.py/ train_doserad_gpu.py use -- READ them and match).
  Configurable d_model, n_layers, n_heads. `.param_count()`. Keep it efficient
  (target < 8M params; DoTA-scale) so epochs are fast.
- scripts/train_doserad_gpu.py -- add "dota3d" as a selectable --model choice (one
  registry line + a --model arg if not present); default stays bragg3d for back-compat.
- tests/test_bt_dota3d.py -- SYNTHETIC CPU-fast: forward shape (B=2,D=32,H=16,W=16),
  nonnegative dose, backward finite grads, param_count in range, and a 2-step loss
  decrease. Plus a real MPS backward check that must not raise.

VERIFY: `.venv/bin/python -m pytest tests/test_bt_dota3d.py -q` and confirm
`scripts/train_doserad_gpu.py --model dota3d --device cpu --fast` (or synthetic path)
runs 2 epochs. Do NOT run git. Print a summary + param count + test result. The
orchestrator runs the real GPU training.
