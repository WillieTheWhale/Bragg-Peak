Implement TWO consensus-confirmed fixes (docs/PLATEAU_DIAGNOSIS.md findings 3 & 4). I have
designed them; you execute exactly as specified. Edit ONLY braggtransporter/data/doserad.py
and scripts/train_doserad_gpu.py. Do NOT run git.

FIX A - depth window clips high-energy Bragg peaks (CAPPING).
The fixed 300mm depth extent cuts off the peak for beamlets up to 200.8 MeV (range ~262mm;
box-entry offset pushes peaks past 300mm). VERIFIED: high-energy beamlets peak at the last
bin with dose still rising.
- In braggtransporter/data/doserad.py: change default `depth_extent_mm` from 300.0 to 400.0
  in extract_bev_pair, DoseRADBeamletDataset.__init__, and make_doserad_loaders. Update the
  cache tag so it reflects de400 (must not collide with de300 caches).
- In scripts/train_doserad_gpu.py: if it passes depth_extent_mm anywhere, set/allow 400;
  add a --depth-extent-mm arg (default 400.0) plumbed to the loaders.
- TEST (tests/test_bt_depth_noclip.py, synthetic sitk): build a beamlet whose dose peak sits
  at ~330mm physical depth; assert with depth_extent_mm=400 the extracted depth profile peak
  is NOT in the last 5 bins AND the profile falls back below 50% of peak after the peak
  (distal falloff captured, i.e. not clipped). Also assert a 300mm window WOULD clip it
  (peak in last bins) to prove the fix matters.

FIX B - same-patient train/val leakage (INFLATES the reported gamma).
random_split over pooled beamlets lets train and val share patients. Make validation a
PATIENT holdout.
- In scripts/train_doserad_gpu.py (and/or make_doserad_loaders): add `--split-by
  {patient,beamlet}` (default patient). In patient mode: choose ceil(val_frac * n_patients)
  whole patients (>=1) by seeded shuffle as the VAL set; ALL their beamlets go to val, all
  other patients' beamlets to train. Guarantee zero patient overlap. If only 1 patient is
  available, fall back to beamlet split and PRINT a clear warning that the number is
  optimistic. Print the chosen val patient IDs.
- TEST (tests/test_bt_patient_split.py): with a synthetic dataset of >=3 patients, patient
  mode yields train/val beamlet sets whose patient ID sets are DISJOINT; beamlet mode may
  overlap. Verify the val patient list is printed/returned.

VERIFY: run `.venv/bin/python -m pytest tests/test_bt_depth_noclip.py tests/test_bt_patient_split.py -q`
and a 2-epoch `--model dota3d_spatial --fast --device cpu --split-by patient` smoke that runs
without error. Print both test results, the new default depth_extent_mm, and the smoke's
chosen val-patient line. Do NOT change the model, WEPL, gamma, or normalization.
