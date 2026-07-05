Implement Phase 6 reproducibility + CI + OOD/abstention for BraggTransporter v3.1.

READ FIRST: PROGRESS.md (all phase results), brag_deep_learning/six-month-model-plan.md
(Phase 6 + §6 three claims A/B/C), braggtransporter/uncertainty.py,
braggtransporter/calibration.py, braggtransporter/metrics.py, scripts/phase*.py.

ONLY create:
- scripts/reproduce.sh — one command that regenerates the KEY laptop-scale results
  end-to-end with pinned seeds: generate the 1-D dataset, train v0, evaluate held-out
  (print gamma/distal-edge), run the Phase-4 calibration, and print a compact
  results table. Use .venv/bin/python -u and PYTHONPATH=. Keep total runtime modest
  (fewer epochs OK — this is a reproducibility smoke, not the full study). Fail (exit
  nonzero) if any key metric regresses past a threshold (e.g. held-out gamma < 90%,
  calibrated coverage_68 outside 0.62-0.74).
- tests/test_bt_regression.py — a FAST CPU regression test asserting the invariants
  that must never break: dose nonneg; range monotonic with energy on a small ladder;
  gamma(curve, curve)=100%; calibration coverage increases monotonically with the
  temperature; v0 back-compat param count unchanged. No MPS, no network.
- scripts/phase6_ood_abstention.py — a small OOD/abstention demo: take the trained
  uncertainty head; show predicted sigma is LARGER on out-of-distribution inputs
  (e.g. energies/materials outside the training range, or corrupted CT) than on
  in-distribution held-out; define an abstention rule (flag beamlets whose mean sigma
  exceeds a calibrated threshold) and report the in-dist vs OOD abstention rates.
  Write docs/results/phase6_ood.csv. Insert repo root into sys.path.

VERIFY: `.venv/bin/python -m pytest tests/test_bt_regression.py -q` and a `bash
scripts/reproduce.sh` dry run at low epochs (report the printed table). The OOD demo:
run against experiments/bt/phase3/braggtransporter_v0/best.pt if present else --fast.
Do NOT run git. Do NOT enable torch.use_deterministic_algorithms on MPS. Print a
summary: the reproduce table, regression test result, and OOD sigma in-dist vs OOD.
