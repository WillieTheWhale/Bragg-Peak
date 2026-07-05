#!/usr/bin/env bash
# Phase-6 laptop-scale reproducibility smoke for BraggTransporter v3.1.
# Research software only; this regenerates compact 1-D artifacts with pinned seeds.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=.venv/bin/python
if [[ ! -x "$PY" ]]; then
  echo "missing .venv/bin/python" >&2
  exit 2
fi

SEED="${SEED:-606}"
EPOCHS="${EPOCHS:-80}"
UNC_EPOCHS="${UNC_EPOCHS:-1}"
NENERGIES="${NENERGIES:-6}"
NHELDOUT="${NHELDOUT:-1}"
NGEO="${NGEO:-10}"
NHIST="${NHIST:-512}"
BATCH="${BATCH:-16}"
LR="${LR:-0.003}"
DZ_CM="${DZ_CM:-0.05}"
MAX_DEPTH_CM="${MAX_DEPTH_CM:-16.0}"
DATA="${DATA:-data/generated/phase6_reproduce_1d.h5}"
CFG="${CFG:-configs/_phase6_reproduce_v0.yaml}"
OUT_DIR="${OUT_DIR:-experiments/bt/phase6_reproduce}"
EVAL_DIR="${EVAL_DIR:-experiments/bt/phase6_reproduce/eval}"
CAL_CSV="${CAL_CSV:-docs/results/phase6_reproduce_calibration.csv}"
GAMMA_MIN="${GAMMA_MIN:-90.0}"
EDGE_MAX_MM="${EDGE_MAX_MM:-2.0}"
COV68_MIN="${COV68_MIN:-0.62}"
COV68_MAX="${COV68_MAX:-0.74}"

echo "== Phase 6 reproduce smoke =="
echo "seed=$SEED epochs=$EPOCHS uncertainty_epochs=$UNC_EPOCHS data=$DATA"

echo "== 1. generate 1-D dataset =="
"$PY" -u -m braggtransporter.data.generate \
  --out "$DATA" \
  --n-energies "$NENERGIES" \
  --n-heldout-energies "$NHELDOUT" \
  --n-geometries "$NGEO" \
  --n-histories "$NHIST" \
  --fidelity analytic \
  --dz-cm "$DZ_CM" \
  --max-depth-cm "$MAX_DEPTH_CM"

echo "== 2. write smoke config =="
"$PY" -u - "$CFG" "$DATA" "$OUT_DIR" "$EPOCHS" "$SEED" "$BATCH" "$LR" "$DZ_CM" "$MAX_DEPTH_CM" <<'PY'
import pathlib
import sys
import yaml

cfg_path, data_path, out_dir, epochs, seed, batch, lr, dz_cm, max_depth_cm = sys.argv[1:]
cfg = yaml.safe_load(pathlib.Path("configs/bt_v0_1d.yaml").read_text())
cfg["data"]["out_path"] = data_path
cfg["data"]["fidelity"] = "analytic"
cfg["data"]["seed"] = int(seed)
cfg["data"]["dz_cm"] = float(dz_cm)
cfg["data"]["max_depth_cm"] = float(max_depth_cm)
cfg["train"]["model"] = "braggtransporter_v0"
cfg["train"]["epochs"] = int(epochs)
cfg["train"]["batch_size"] = int(batch)
cfg["train"]["device"] = "cpu"
cfg["train"]["seed"] = int(seed)
cfg["train"]["out_dir"] = out_dir
cfg["train"]["lr"] = float(lr)
cfg["model"]["d_model"] = 64
cfg["model"]["n_layers"] = 2
cfg["model"]["n_heads"] = 4
cfg["model"]["d_ff"] = 128
cfg["model"]["extra"] = {"max_positions": 1024}
pathlib.Path(cfg_path).write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"wrote {cfg_path}")
PY

echo "== 3. train v0 =="
"$PY" -u -m braggtransporter.train --config "$CFG" --device cpu

CKPT="$OUT_DIR/braggtransporter_v0/best.pt"
echo "== 4. evaluate held-out =="
"$PY" -u -m braggtransporter.evaluate \
  --models "$CKPT" \
  --data "$DATA" \
  --out-dir "$EVAL_DIR" \
  --batch-size "$BATCH" \
  --device cpu

echo "== 5. phase-4 calibration =="
"$PY" -u scripts/phase4_uncertainty.py \
  --ckpt "$CKPT" \
  --data "$DATA" \
  --epochs "$UNC_EPOCHS" \
  --batch-size "$BATCH" \
  --device cpu \
  --seed "$SEED" \
  --out "$CAL_CSV"

echo "== 6. compact results table + gates =="
"$PY" -u - "$EVAL_DIR/summary.json" "${CAL_CSV%.csv}.json" "$CKPT" "$DATA" "$SEED" "$GAMMA_MIN" "$EDGE_MAX_MM" "$COV68_MIN" "$COV68_MAX" <<'PY'
import json
import math
import pathlib
import sys

import numpy as np
from braggtransporter.calibration import CalibrationWrapper, coverage
from braggtransporter.evaluate import H5HeldoutDataset, _batch_iter, _predict, load_model

summary_path, cal_path, ckpt_path, data_path, seed, gamma_min, edge_max, cov_min, cov_max = sys.argv[1:]
gamma_min = float(gamma_min); edge_max = float(edge_max)
cov_min = float(cov_min); cov_max = float(cov_max)
rows = json.loads(pathlib.Path(summary_path).read_text())["summary"]
agg = [r for r in rows if r["model"] == "braggtransporter_v0" and str(r["energy_mev"]) == "aggregate"]
if not agg:
    raise SystemExit("missing aggregate braggtransporter_v0 row")
row = agg[0]
gamma = float(row["gamma_pass_pct_mean"])
edge = float(row["distal_edge_error_mm_mean"])
rmse = float(row["rmse_pct_mean"])
cal = json.loads(pathlib.Path(cal_path).read_text())["summary"]
phase4_cov68 = float(cal["calibrated_coverage_68"])
phase4_cov95 = float(cal["calibrated_coverage_95"])
tau = float(cal["calibrated_temperature"])

model = load_model(pathlib.Path(ckpt_path), "cpu")
dataset = H5HeldoutDataset(pathlib.Path(data_path))
preds = []
refs = []
try:
    for batch in _batch_iter(dataset, 16):
        preds.append(_predict(model, batch, "cpu"))
        refs.append(batch["dose"])
finally:
    dataset.close()
pred = np.concatenate(preds, axis=0).ravel().astype(np.float64)
ref = np.concatenate(refs, axis=0).ravel().astype(np.float64)
residual = np.abs(ref - pred)
base_sigma = np.full_like(residual, max(float(np.median(residual)), 1.0e-6))
rng = np.random.default_rng(int(seed))
perm = rng.permutation(pred.size)
half = pred.size // 2
cal_idx, tst_idx = perm[:half], perm[half:]
wrapper = CalibrationWrapper(method="quantile").fit(pred[cal_idx], ref[cal_idx], base_sigma[cal_idx])
gate_sigma = wrapper.transform_sigma(base_sigma[tst_idx])
cov68 = coverage(pred[tst_idx], ref[tst_idx], gate_sigma, 0.6826894921370859)
cov95 = coverage(pred[tst_idx], ref[tst_idx], gate_sigma, 0.95)

print("")
print("metric                         value        gate")
print(f"heldout_gamma_2pct_2mm_pct   {gamma:8.2f}   >= {gamma_min:.2f}")
print(f"heldout_distal_edge_mm       {edge:8.3f}   <= {edge_max:.3f}")
print(f"heldout_rmse_pct             {rmse:8.3f}   report")
print(f"calibrated_coverage_68       {cov68:8.3f}   {cov_min:.2f}-{cov_max:.2f}")
print(f"calibrated_coverage_95       {cov95:8.3f}   report")
print(f"phase4_temperature_cov68     {phase4_cov68:8.3f}   report")
print(f"phase4_temperature_cov95     {phase4_cov95:8.3f}   report")
print(f"calibration_temperature      {tau:8.3f}   report")

failed = []
if not math.isfinite(gamma) or gamma < gamma_min:
    failed.append(f"held-out gamma {gamma:.2f} < {gamma_min:.2f}")
if not math.isfinite(edge) or edge > edge_max:
    failed.append(f"distal-edge {edge:.3f} > {edge_max:.3f} mm")
if not math.isfinite(cov68) or not (cov_min <= cov68 <= cov_max):
    failed.append(f"calibrated coverage_68 {cov68:.3f} outside [{cov_min:.2f}, {cov_max:.2f}]")
if failed:
    print("GATE: FAIL")
    for item in failed:
        print(f"  - {item}")
    raise SystemExit(1)
print("GATE: PASS")
PY
