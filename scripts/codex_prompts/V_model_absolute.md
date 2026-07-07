Fix CONFIRMED bug 1 in braggtransporter/models/dota3d_spatial.py ONLY. Context in
docs/PLATEAU_DIAGNOSIS.md: the model's `slice_norm = nn.GroupNorm(1, c_in)` REMOVES the
absolute CT/density level, so a water slab [0,1,1] and a dense slab [1,2,2] become identical
tensors -> the model is blind to tissue stopping power (which sets Bragg-peak depth). VERIFIED.
Do NOT touch data or trainer files.

SHARED CONTRACT (the data file is being edited in parallel to produce this EXACTLY):
- Input x now has 4 channels: [hu_over_1000, density, rsp, wepl_norm] (RAW physical values,
  NOT pre-normalized). c_in defaults to DOSERAD_INPUT_CHANNELS (import it; it will be 4).

FIX: replace the per-slice GroupNorm that erases absolute level with FIXED per-channel
standardization that PRESERVES absolute differences:
- Add a registered buffer `input_mean` and `input_std` of shape (1, c_in, 1, 1) with sensible
  fixed constants for [hu/1000~0.0/0.6, density~1.0/0.4, rsp~1.0/0.4, wepl~0.5/0.3] (approx;
  pick reasonable fixed values). Standardize slices as (x - input_mean)/input_std. This keeps
  water vs bone DISTINCT (unlike GroupNorm). Do NOT use GroupNorm/InstanceNorm/LayerNorm over
  the channel+spatial dims for the input. (LayerNorm on d_model tokens elsewhere is fine.)
- Confirm the model still accepts variable depth D and the 4-channel input; keep the forward
  contract forward(x:(B,4,D,H,W), scalars:(B,4)) -> {"dose":(B,D,H,W)}, softplus, .param_count().

CRITICAL TEST (tests/test_bt_spatial_absolute.py, CPU-fast):
- The OLD failure must be GONE: build two homogeneous inputs, a water slab (hu=0,dens=1,rsp=1,
  wepl ramp) and a dense slab (hu=1,dens=1.9,rsp=1.6, wepl steeper ramp), same energy/gantry.
  Assert the model outputs DIFFER meaningfully (mean abs diff > 1e-3) -> model now sees density.
- forward shape (B=2,4,D=16,H=24,W=24)->dose(2,16,24,24), nonnegative, finite grads, param_count<8M.
VERIFY: `.venv/bin/python -m pytest tests/test_bt_spatial_absolute.py -q`. Do NOT run git.
Print the water-vs-dense output difference (must be >> 1e-3) and test results.
