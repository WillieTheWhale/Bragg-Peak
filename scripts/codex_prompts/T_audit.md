ADVERSARIAL CORRECTNESS AUDIT of a 3-D proton dose-prediction pipeline. This code was
written carefully, passes its tests, and is BELIEVED CORRECT. Your job is NOT to please a
reviewer and NOT to invent problems. Your DEFAULT POSITION must be "this code is correct."

Rules to avoid false positives:
- For every candidate issue, FIRST write the strongest argument that it is INTENTIONAL and
  CORRECT. Only if you can then prove a concrete, specific failure that survives that
  defense may you flag it. If you cannot construct a defeating failure case, do NOT flag it.
- A valid finding MUST include a concrete failure scenario (specific input/state -> wrong
  output) and WHY it caps accuracy. Vague "could be improved" is not allowed.
- Rank by severity. It is completely acceptable to conclude a file has NO real bugs.
- Do NOT edit code. Do NOT run git. Report only.

Context (do NOT treat as hints toward specific bugs — audit everything even-handedly):
we train a beam's-eye-view depth model on DoseRAD2026 proton beamlets and get ~69% gamma
(3%/3mm); the reference papers reach ~99% (1%/3mm). The gap may be pure data scale/model,
OR a real pipeline bug. Determine which, honestly.

Audit these for correctness bugs that would corrupt the input->dose problem or cap accuracy:
- braggtransporter/data/doserad.py: extract_bev_pair (BEV geometry: ray axis, ray-box
  entry, lateral axes, output origin/direction, resampling default values), hu_to_density_rsp
  (HU->density and HU->RSP for proton stopping power), parse_plan / _iter_patient_records
  (does each dose .mha get matched to the correct beam/ray/energy/gantry/source/target?),
  normalize_beamlet_dose + dose_scale usage.
- scripts/train_doserad_gpu.py: the regression loss (target, any masking/weighting), the
  train/val split, the scalars fed to the model, evaluate() (pred vs ref in the same space?).
- braggtransporter/models/dota3d_spatial.py: can the energy/gantry scalars actually place
  the Bragg peak by depth, or is that information lost?

Output: a ranked list of CONFIRMED bugs (each with file:line, the failed defense, the
concrete failure scenario, severity, fix), then a short list of things you checked and
judged CORRECT (so we know coverage). Be willing to tell me nothing is wrong.
