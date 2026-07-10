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

## run18 final result (legacy protocol; not paper-comparable)

Run18 completed 150 epochs on 2026-07-10 at source commit
`900e31151d467ba1cd08745c90046e3448683db9`. The resumed run used the
iteration-1 protocol: 12 patients, a lexicographic 500-beamlet prefix per
patient, patient holdout with validation patients `1ABB041` and `1ABB020`,
per-beamlet unit-max relative dose, a 201-bin/400mm depth axis (2.0mm voxel
spacing), and a 24-bin/96mm lateral grid (4.174mm spacing).

The gamma-selected checkpoint was epoch 138:

- Internal 96-beamlet 3%/3mm/10%-cutoff gamma: 84.79%.
- Full 1,000-beamlet 3%/3mm/10%-cutoff gamma: 86.24%.
- Full RMSE: 2.26%; full mean R80 error: 1.10mm.
- Full gamma progression: 80.56% (epoch 40), 84.66% (epoch 80),
  83.13% (epoch 120), and 86.24% for the frozen best checkpoint.

These results show that the 2.0mm depth grid removed a major accuracy cap, but
they remain an internal diagnostic. Run18 did not use the all-angle stratified
sample, an untouched test cohort, or DoTA's 1%/3mm/0.1%-cutoff criterion.
No comparison with the paper's headline gamma should use 86.24%.

The complete repaired artifact set is under
`gs://braggtransporter-braggtransporter/runs/run18/`: 150-row CSV and JSONL,
NPZ arrays, best-full JSON, metadata/audit JSON, checkpoints, and terminal logs.

## run19 launch record

Run19 launched on 2026-07-10 from `audit-iter2` commit
`94c7dd4fe4790d54e1ab55015fb0670ac92e60de` on an on-demand L4
(`g2-standard-12`, `us-central1-b`). It keeps run18's model, seed, physical
grid, optimizer, and 150-epoch schedule, with these controlled protocol changes:

- deterministic 500-beamlet patient manifests spanning all 36 beams
  (13-14 samples per beam, verified for all 12 patients);
- `--test-frac 0.15 --val-frac 0.08`, yielding 9 train, 1 validation, and 2
  untouched test patients;
- full/final reporting at 3%/3mm/10%, 2%/2mm/10%, and the DoTA beamlet
  criterion of 1%/3mm with a 0.1% cutoff.
