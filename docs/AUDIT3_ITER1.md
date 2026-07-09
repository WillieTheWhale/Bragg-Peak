# Double-blind audit #3, iteration 1 (2026-07-09)

Protocol: Claude (this repo's operator agent) and a GPT-5.6 Sol session
(`codex exec -m gpt-5.6-sol`, read-only sandbox) audited the pipeline
independently. The Sol prompt (`scripts/codex_prompts/Y_audit3_iter1.md`) put
the burden of proof on the auditor: default verdict "working as intended",
findings require a concrete reproducible mismatch between stated intent and
actual behavior. Neither auditor saw the other's findings before both reported.

## Consensus finding (both auditors, independently): the resolved grid silently
## degraded resolution AND made the gamma DTA term inert

run17's launcher passed `--depth-extent-mm 400` but not `--depth-size`, so the
argparse default of **64** applied. Resolved grid: depth 400/(64-1) =
**6.35mm** bins (intent, per commit 9743684 / PROGRESS.md: ~2.4mm), lateral
96/(24-1) = **4.17mm** bins (papers: 2mm).

Proof points (all reproduced empirically):
- Checkpoint of record: `runs/dota15/best.pt` saved `args.depth_size = 64`,
  `depth_extent_mm = 400.0` → 6.349mm depth spacing (Claude, from GCS).
- At 6.35mm depth / 4.17mm lateral spacing, EVERY nonzero voxel displacement
  exceeds the 3mm DTA: minimum normalized distance term is (6.35/3)^2 = 4.48
  in depth, (4.17/3)^2 = 1.93 laterally. The voxel-center gamma search can
  only ever pass on the same voxel → "gamma 3%/3mm" degenerated to a
  same-voxel 3% dose-difference test (Sol: synthetic 9-point profile with a
  ~1% dose offset scores 0.00% repo-gamma vs 88.89% interpolated gamma).
- A 2mm physical depth shift — fully inside the papers' 3mm DTA tolerance —
  passes 100% on a 2.0mm grid but only ~86% on the 6.35mm grid and ~85% on a
  3.15mm grid (Claude, 12 real beamlets, 4 patients). Our headline number is
  therefore HARSHER than the papers' metric, i.e. non-comparable in the
  conservative direction.
- WEPL input channel quantization at 64 bins: mean 2.8mm / max 4.4mm
  central-axis error vs a 2mm grid (Claude, 12 real beamlets) — the depth
  localization signal fed to the model is corrupted by more than the 3mm DTA
  criterion itself.
- Falloff (R80→R20) width reads 6.86mm at 64 bins vs 6.15mm at 2mm bins:
  the coarse grid smears the distal edge the loss and metric care about.

Independent verdicts agreed; also independently agreed as VERIFIED-CORRECT:
patient-holdout split (zero overlap), best-checkpoint full-eval procedure,
dose normalization + 10% low-dose mask, HU→density/RSP distinct curves,
no target leakage into the model, 400mm extent eliminates peak clipping.

## Consensus fix (run18, branch audit-iter1)

1. Launch with explicit `--depth-size 201` → 400/(201-1) = **2.0mm** depth
   bins, matching the papers' grid on the beam axis and making the 3mm DTA
   meaningful in depth. (Lateral 2mm — `--lateral-size 49` — quadruples the
   token count and is deferred to a later iteration as a separate lever.)
2. New parameterized launcher `scripts/gcp_iter_launch.sh` (run name, git
   branch, GPU type, train args are arguments — removes the class of
   hand-edited-constant errors that caused run17 to log into run15's GCS
   path and silently inherit defaults).
3. `train_doserad_gpu.py` now prints the resolved grid spacing at startup and
   warns loudly when spacing exceeds the 3mm DTA (metric-inertness guard).
4. Regression test: a 2mm sub-DTA shift must pass gamma on a 2mm grid and
   must fail on run17's 6.35mm grid
   (`tests/test_bt_doserad_physics.py::test_gamma_dta_grants_credit_on_2mm_grid_but_not_on_coarse_grid`).

Expected effect: honest gamma becomes paper-comparable in depth (DTA active),
the model sees an uncorrupted WEPL/depth signal (≤2mm quantization), and the
distal edge is representable. Cost: ~3.1x depth tokens; run18 uses L4 spot
(fallback T4) to keep 150 epochs feasible.

## Sol's residual open items (queued for later iterations)
- Lateral spacing 4.17mm still leaves the DTA inert laterally (metric remains
  conservative vs papers on that axis).
- HU→density/RSP calibration curves are generic, not DoseRAD-scanner-specific.
- Downloader takes the FIRST 500 plan-order beamlets per patient (gantry-angle
  truncation bias), not a seeded random sample.
- Best-epoch selection uses a 96-beamlet subsample of the same held-out
  patients later used for the headline number (no independent test split).
