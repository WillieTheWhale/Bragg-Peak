VERIFICATION AUDIT #3, iteration 2, of a 3-D proton dose pipeline. The pipeline passes
tests and is BELIEVED to be WORKING AS INTENDED. Your job is NOT to "find errors."
For every part you examine, either (a) demonstrate with CONCRETE PROOF that it does
exactly what the operators intend and claim, or (b) prove with a concrete, reproducible
scenario that it is NOT working as intended. The default verdict for anything you cannot
prove either way is "UNVERIFIED", not "bug".

Burden-of-proof rules (same as iteration 1):
- For every candidate deviation, FIRST write the strongest argument that the behavior is
  INTENTIONAL/CORRECT. Only report it if a CONCRETE demonstration (numbers, real data
  from ./data/doserad2026, or a runnable snippet) survives that defense.
- "Suboptimal design" is not a finding. A finding is a proven mismatch between claimed/
  intended behavior and actual behavior, or something that makes the headline metric
  dishonest/unrepresentative, or an unrecognized hard cap on achievable accuracy.
- Read-only: you may run python/git inspection commands, but no edits, no training.
- Concluding "everything checked is working as intended" is a good outcome if true.

Context:
- Task: predict BEV proton beamlet dose on DoseRAD2026 (public). Papers (DoTA/ADoTA)
  report ~99% gamma (1%/3mm, 2%/2mm) using ~80k beamlets across ~55+ patients.
- Iteration 1 already found and fixed a grid-resolution/gamma-DTA flaw (see
  docs/AUDIT3_ITER1.md). run18 is now training with 2.0mm depth bins. Do NOT re-report
  that finding or its disclosed residuals as your primary result unless you can prove
  something NEW about them.
- Config of record for the in-flight run18 (branch audit-iter1):
  scripts/gcp_iter_launch.sh --run-name run18 --gpu l4 --train-args
    "--epochs 150 --batch-size 8 --model dota3d_spatial --d-model 192 --n-layers 6
     --lr 3e-4 --split-by patient --depth-size 201 --depth-extent-mm 400
     --allow-coarse-axes lateral --lr-schedule dota --restart-epochs 28
     --weight-decay 0.1 --eval-subsample 96 --full-eval-every 40
     --checkpoint-every-steps 0"
  with 12 patients, 500 beamlets per patient (scripts/vm_download_doserad.py).
- Operator intent this iteration verifies: (i) the training and validation beamlets are
  a FAITHFUL, REPRESENTATIVE sample of the DoseRAD2026 proton beamlet distribution, so
  the held-out number generalizes to "beamlet dose prediction on this dataset" the way
  the papers' numbers do; (ii) the objective/loss trains the quantity the metric
  evaluates; (iii) the training/selection/reporting protocol produces an honest,
  paper-comparable headline number; (iv) the run17->run18 changes did not introduce
  regressions.
- Local real data for proof: ./data/doserad2026 has 4 patients (ct.mha, plan.json,
  ~45 dose files each). plan.json structure: beams[] (beam_idx, gantry_angle, rays[]
  (ray_idx, ray_source, ray_target, beamlets[] (beamlet_idx, energy))).

Files in scope (read anything else you need):
1. scripts/vm_download_doserad.py   — what lands on the VM and in what order.
2. braggtransporter/data/doserad.py — record selection, caching, split, loaders.
3. scripts/train_doserad_gpu.py     — loss/objective, eval protocol, best-epoch
                                      selection, subsample construction, schedules.
4. scripts/gcp_iter_launch.sh       — launch/data flow for run18.

Report format:
- CONFIRMED FINDINGS (ranked): file:line, violated intent, failed defense, concrete
  proof with numbers, severity, minimal fix.
- VERIFIED AS INTENDED: one line each.
- UNVERIFIED: what you could not prove either way.
