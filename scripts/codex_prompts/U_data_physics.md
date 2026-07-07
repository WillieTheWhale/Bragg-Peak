Fix two CONFIRMED pipeline bugs in braggtransporter/data/doserad.py ONLY. Context in
docs/PLATEAU_DIAGNOSIS.md (consensus bugs 1 & 2). Do NOT touch model or trainer files.

SHARED CONTRACT (the model file is being edited in parallel to match this EXACTLY):
- DOSERAD_INPUT_CHANNELS = 4, channel order: [hu_over_1000, density, rsp, wepl_norm].
- Channels are RAW PHYSICAL values in the cache (NO per-slice normalization here; the model
  applies fixed standardization). density and rsp MUST differ (real conversions).
- wepl_norm[d,y,x] = (cumulative sum over depth d of rsp[:d,y,x] * depth_spacing_mm) / 300.0
  (water-equivalent path length; the physics prior that tells where the Bragg peak lands).
- Cache each record's `depth_spacing_mm` (float) and keep `spacing_mm` (3-vector) as before.

FIX 1 (CRITICAL - material): replace hu_to_density_rsp with a REAL piecewise conversion:
- density_g_cm3(HU): stepwise/linear Schneider-like segments (air ~0.0 at HU<=-950, lung,
  adipose/water ~1.0 near HU 0, soft tissue, bone rising to ~1.9 by HU~1600), clamped [0,2.2].
- rsp(HU): a SEPARATE proton relative-stopping-power curve (NOT equal to density): ~0 in air,
  ~1.0 in water (HU 0), rising to ~1.6-1.7 in dense bone. Piecewise-linear is fine.
- Keep the function signature returning (density, rsp) but make them genuinely different.

FIX 2 (HIGH - depth scale): in extract_bev_pair, make the depth grid a FIXED physical extent
so spacing is CONSISTENT across beamlets/patients: add param `depth_extent_mm: float = 300.0`;
place the depth axis from the CT entrance along the ray spanning depth_extent_mm at
depth_spacing = depth_extent_mm/(depth_size-1) (so 128 bins -> ~2.36mm; pass depth_size from
the dataset, default it to 128). Build the [hu,density,rsp] then append the wepl_norm channel
computed from rsp along depth. Return x with shape (4, depth, H, W) and the depth_spacing_mm.

Update DoseRADBeamletDataset._write_cache to store the 4-channel x, wepl already included,
plus depth_spacing_mm in the npz; bump DOSERAD_INPUT_CHANNELS=4; default depth_size=128 in
the dataset + extract_bev_pair + make_doserad_loaders. Keep the cache tag reflecting the new
channels/spacing so old caches don't collide (e.g. add 'c4_wepl' to the tag).

TESTS (tests/test_bt_doserad_physics.py, CPU-fast, synthetic sitk images):
- density != rsp for a bone-like HU; both ~1.0 at HU 0; air ~0.
- x has 4 channels; wepl is monotonically non-decreasing along depth and 0 at entrance.
- depth_spacing_mm is constant across two different synthetic rays (fixed extent).
VERIFY: `.venv/bin/python -m pytest tests/test_bt_doserad_physics.py -q`. Do NOT run git.
Print DOSERAD_INPUT_CHANNELS, a sample density-vs-rsp pair, and test results.
