VERIFICATION AUDIT #3 (iteration 1) of a 3-D proton dose pipeline. The pipeline passes
tests and is BELIEVED to be WORKING AS INTENDED. Your job is NOT to "find errors."
Your job is: for every part of the system you examine, either (a) demonstrate with
CONCRETE PROOF that it is doing exactly what the operators intend and claim, or
(b) prove with a concrete, reproducible failure scenario that it is NOT working as
intended. Claims without proof in either direction are worthless — the default
verdict for anything you cannot prove either way is "UNVERIFIED", not "bug".

Burden-of-proof rules:
- For every candidate deviation you consider, FIRST write the strongest argument that
  the behavior is INTENTIONAL and CORRECT. Only report it if you can then produce a
  CONCRETE demonstration (specific config value, input, or state -> specific unintended
  behavior, with numbers) that survives that defense.
- "Suboptimal design" is not a finding. A finding must be a mismatch between what the
  system is CLAIMED/INTENDED to do and what it ACTUALLY does, or something that makes
  the reported headline metric dishonest/non-comparable, or something that concretely
  caps achievable accuracy in a way the operators are unaware of.
- You may run read-only commands (python to inspect files/arrays, git log/show) to
  gather proof. Do NOT edit code, do NOT run git write operations, do NOT train.
- It is a perfectly good outcome to conclude "everything checked is working as intended."

Context (the intent you are verifying against):
- Task: predict beam's-eye-view (BEV) proton beamlet dose on the public DoseRAD2026
  dataset. Reference papers (DoTA / ADoTA, 2022-2026) report ~99% gamma at 1%/3mm and
  2%/2mm on ~80k beamlets with 2mm voxel grids and a ~3M-param transformer.
- Our latest completed cloud run ("run17") is the CONFIG OF RECORD, launched by
  scripts/gcp_scaling_startup.sh, which executed exactly:
    scripts/train_doserad_gpu.py --patients <12 patients> --max-beamlets 500
      --epochs 150 --device cuda --batch-size 10 --model dota3d_spatial
      --d-model 192 --n-layers 6 --lr 3e-4 --split-by patient --depth-extent-mm 400
      --lr-schedule dota --restart-epochs 28 --weight-decay 0.1 --eval-subsample 96
      --full-eval-every 40
  (any argument not listed took the argparse default in scripts/train_doserad_gpu.py)
- The operators' stated intent for this run (from commit messages / progress log):
  (i) physics-fixed inputs: real HU->density/RSP + WEPL channel, and a FIXED,
  CONSISTENT, FINE depth grid ("~2.4mm depth bins") so the model sees true depth scale;
  (ii) 400mm depth extent so no high-energy Bragg peak is clipped; (iii) an HONEST
  held-out number: patient-holdout split, standard global 3-D gamma at 3%/3mm with 10%
  low-dose threshold, computed on the SAME grid the papers would use it on, no leakage,
  no inflated metric. Result was 77.65% full-eval gamma on 2 held-out patients.
- Goal of this audit iteration: find at most the ONE OR TWO most consequential proven
  mismatches between intent and actual behavior (in data organization, data download,
  BEV extraction, training configuration, model, or evaluation) — the things most
  likely to explain part of the 77.65% vs ~99% gap or to make 77.65% dishonest.

Files in scope (read anything else you need for proof):
1. scripts/gcp_scaling_startup.sh          — the launcher; config of record.
2. scripts/train_doserad_gpu.py            — argparse defaults, split, loaders, loss,
                                             eval, gamma driver, checkpointing.
3. braggtransporter/data/doserad.py        — download, plan parsing, beamlet selection,
                                             BEV extraction geometry, WEPL, caching,
                                             normalization, split, gamma implementation.
4. braggtransporter/models/dota3d_spatial.py — model, standardization, decoding.
5. scripts/vm_download_doserad.py          — what data actually lands on the VM.

Report format:
- CONFIRMED FINDINGS (ranked by consequence): for each — file:line, the intent it
  violates, the failed defense, the concrete proof (numbers), severity, and the
  minimal fix you would make.
- VERIFIED AS INTENDED: what you checked and proved correct (one line each).
- UNVERIFIED: what you could not prove either way.
