Fix an MPS-compatibility bug in `braggtransporter/models/braggtransporter_v0.py`.

PROBLEM: the coordinate-query decoder uses `torch.nn.functional.grid_sample`
(grid_sampler_2d). Its backward pass `aten::grid_sampler_2d_backward` is NOT
implemented on the Apple MPS device, so training on MPS crashes with
`NotImplementedError`. We train on MPS (Apple M4 Max) — a CPU fallback is too slow.

FIX: replace grid_sample with a **pure-torch, MPS-native, differentiable 1-D
linear interpolation** along the depth axis. The decoder queries per-depth latent
features (and/or predicted quantities) at arbitrary normalized depth coordinates
`z_query in [0,1]`. Implement linear interpolation with `torch.gather` / index math
and clamping (no grid_sample, no torch ops that lack MPS backward). Keep the exact
public behavior: same `forward(x,scalars)->dict` signature and same default
(on-grid) outputs; keep the optional `z_query` keyword if it exists.

ONLY edit:
- `braggtransporter/models/braggtransporter_v0.py`
- `tests/test_bt_v0_interp.py`  (new)

Do NOT edit schema.py/config.py or any other module.

Add `tests/test_bt_v0_interp.py`:
- forward + `loss.backward()` runs WITHOUT error on CPU and produces finite grads;
- off-grid coordinate query returns finite values with the right shape;
- the new interpolation matches a reference numpy linear interpolation within 1e-4
  on a known latent field;
- a 2-step optimization step reduces a simple dose MSE loss.

Verify with `.venv/bin/python -m pytest tests/test_bt_v0.py tests/test_bt_v0_interp.py -q`
AND run a real MPS check:
`PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python - <<'PY'
import torch; from braggtransporter.models.braggtransporter_v0 import BraggTransporterV0
from braggtransporter.config import ModelConfig, get_device
d=get_device('auto'); m=BraggTransporterV0(ModelConfig()).to(d)
x=torch.randn(2,64,9,device=d); s=torch.randn(2,4,device=d)
out=m(x,s); out['dose'].sum().backward(); print('MPS backward OK on', d)
PY`
Both must pass (the MPS check must NOT raise a NotImplementedError). Do NOT run git.
Print a short summary + the MPS check result.
