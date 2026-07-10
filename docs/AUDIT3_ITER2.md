# Double-blind audit #3, iteration 2 (2026-07-09)

Protocol as iteration 1 (docs/AUDIT3_ITER1.md): independent primary and secondary
audits using the burden-of-proof prompt in `scripts/codex_prompts/Z_audit3_iter2.md`;
findings were compared only after both reports were complete.

## Consensus finding A (found INDEPENDENTLY by both): angular truncation of the
## beamlet sample

`vm_download_doserad.py` took `listing[:500]` per patient — a lexicographic prefix.
Every DoseRAD proton plan is 36 beams at uniform 10° gantry steps × 30 beamlets
(1080 total). The 500-prefix covers only **17/36 angles** (52.8% of angular support
omitted) — verified identically on all 4 local patients by both auditors. The
secondary audit
additionally showed the omitted directions traverse systematically different
anatomy (patient 1ABB021 central-axis WEPL: covered angles mean 197.9mm/max 334.8mm
vs omitted 208.8mm/max 367.5mm), so BEV rotation-invariance does not excuse the
omission. Both train AND val inherited the same bias: the headline measured 17/36
of the distribution.

Fix: `stratified_beamlet_paths()` — deterministic round-robin across beams
(ray/layer order within beam), manifest written to `<patient>/manifest.json`.
Regression: `test_stratified_selection_spans_all_beams` (all 36 beams, ±1 count).

## Consensus finding B: the headline gamma
## was not the papers' metric

We reported only 3%/3mm with a 10% low-dose cutoff. DoTA/ADoTA's beamlet headline
is **1%/3mm with a 0.1% cutoff** (2%/2mm/10% is their accumulated-plan metric).
Counterexamples: a uniform 2.5% underdose scores 100% on our metric, 0% on both
paper criteria; a beamlet missing every 5%-dose tail voxel scores 100% on ours,
0.8% on DoTA's. Numbers were mutually non-comparable in BOTH directions.

Fix: `evaluate()` takes a criteria list; full/final evals report
`gamma3d_1pct_3mm_dota` (0.1% cutoff), `gamma3d_2pct_2mm` (10%), and the legacy
`gamma3d_3pct_3mm` (10%, labeled internal-diagnostic). Made affordable by
`gamma_index_3d_fast` (offset-vectorized, pass/fail-identical to the reference
implementation; regression `test_fast_gamma_matches_reference_pass_rate`).
Per-epoch selection still uses cheap 3%/3mm (internal only).

## Consensus finding C: no untouched test
## cohort

Best-epoch selection (96-beamlet subsample) and the headline number used the SAME
two val patients. A null simulation measured ~0.75pp selection inflation even
for equally-good epochs; ADoTA uses an independent test cohort.

Fix: `--test-frac` creates an untouched TEST patient cohort = the FIRST slice of
the seeded shuffle — deliberately the same patients (1ABB041, 1ABB020 at seed 0)
that run17/run18 used as val, so headline numbers remain comparable across runs.
Selection moves to a separate val patient; test is evaluated exactly once, on the
final frozen best checkpoint, with all three gamma criteria
(`HEADLINE (untouched test patients...)` line + metrics_test.json).

## Also adopted: launch hardening
Launcher now supports `--expect-sha`: the VM asserts the cloned commit matches
before training (a missing `--branch` would otherwise silently train `main`).

## run19 config delta vs run18
Same model/schedule/grid (2.0mm depth × 201). New: stratified 500/patient manifest
(all 36 beams), `--test-frac 0.15 --val-frac 0.08` (train 9 / val 1 / test 2
patients), paper-criteria reporting. Note: run19's 3%/3mm number is expected to
DROP vs run18 even if the model improves — the eval now includes the harder
omitted-angle beamlets and the honest protocol removes selection bias. The
paper-comparable number is `test_gamma3d_1pct_3mm_dota`.
