ADVERSARIAL CORRECTNESS AUDIT #2 of a 3-D proton dose pipeline. The code passes tests and
is BELIEVED CORRECT. Your DEFAULT POSITION is "this code is correct." Do NOT invent problems.

Rules (to avoid false positives):
- For every candidate issue, FIRST write the strongest argument it is INTENTIONAL/CORRECT.
  Only flag it if you can then prove a CONCRETE failure (specific input/state -> wrong output)
  that survives that defense. If you cannot, do NOT flag it.
- Rank by severity. It is fine to conclude a file is correct. Do NOT edit code or run git.

Context: we predict beam's-eye-view proton beamlet dose (DoseRAD2026). Current held-out
gamma is 87% at 3%/3mm on ~6000 beamlets; the reference paper reports ~99% at 1%/3mm.
The gap may be data scale / metric strictness, OR remaining pipeline bugs. Determine which.

Audit ALL of these even-handedly for CORRECTNESS bugs that would cap accuracy OR make our
reported gamma NON-COMPARABLE to the paper (i.e. inflated/optimistic):
1. scripts/train_doserad_gpu.py -- the train/val split: are train and validation beamlets
   allowed to come from the SAME patients (data leakage inflating val gamma)? The loss and
   target; the subsample-vs-full eval; best-checkpoint selection; anything that would make
   the held-out number optimistic or not reflect true generalization.
2. braggtransporter/data/doserad.py -- gamma_index_3d: is it a standard global gamma? the
   low_dose_threshold (10%), per-beamlet max normalization, local search radius, and whether
   3%/3mm here is measured the SAME way papers measure beamlet gamma (global vs local, dose
   normalization, which points are included). Would our convention read HIGHER than a
   standard 1%/3mm or 3%/3mm gamma? extract_bev_pair: does the fixed 300mm depth extent CLIP
   the Bragg peak for the highest-energy beamlets (range near/over 300mm)? Is WEPL
   (cumulative sum of rsp*spacing) computed correctly and leak-free (no using future depth)?
   hu_to_density_rsp curves sane?
3. braggtransporter/models/dota3d_spatial.py -- input standardization buffers, patch grid,
   depth handling for variable D, decoder upsampling: any information leak (e.g. the model
   seeing the target), or any path that trivially inflates train/val agreement.

For each CONFIRMED bug: file:line, the failed defense, concrete failure scenario, severity,
fix. Then list what you checked and judged CORRECT. Distinguish "caps accuracy" from "makes
our gamma non-comparable/inflated" -- both matter. Be willing to say nothing is wrong.
