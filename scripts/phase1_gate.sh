#!/usr/bin/env bash
# Phase-1 GATE for BraggTransporter v3.1.
# PASS iff: (1) all module tests green, and (2) BraggTransporter-v0's held-out
# mean distal-edge error (mm) is <= every baseline (MLP, FNO1d, DoTA) — the
# empirical form of the "attention beats FNO at the sharp distal edge" claim.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
EPOCHS="${EPOCHS:-40}"
DATA="data/generated/phase1_1d.h5"

echo "== 1. unit tests =="
$PY -m pytest tests/test_bt_data.py tests/test_bt_models_baseline.py \
              tests/test_bt_v0.py tests/test_bt_metrics.py -q

echo "== 2. generate 1-D dataset =="
# Phase-1 Stage A predicts the SMOOTH MEAN, so it trains on clean (analytic)
# targets; the stochastic SDE residual is the Phase-4 flow head's job. Clean
# targets also isolate the architecture claim (sharp distal edge) from noise.
NGEO="${NGEO:-40}"; FID="${FID:-analytic}"; NHIST="${NHIST:-4000}"
$PY -m braggtransporter.data.generate --out "$DATA" \
    --n-geometries "$NGEO" --n-histories "$NHIST" --fidelity "$FID"

echo "== 3. train v0 + baselines =="
for m in braggtransporter_v0 mlp fno1d dota; do
  cfg="configs/_gate_${m}.yaml"
  $PY - "$m" "$cfg" "$EPOCHS" <<'PYEOF'
import sys, yaml, pathlib
m, cfg, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3])
base = yaml.safe_load(pathlib.Path("configs/bt_v0_1d.yaml").read_text())
base.setdefault("train", {})["model"] = m
base["train"]["epochs"] = epochs
pathlib.Path(cfg).write_text(yaml.safe_dump(base))
print("wrote", cfg, "model=", m, "epochs=", epochs)
PYEOF
  echo "-- training $m --"
  $PY -m braggtransporter.train --config "$cfg"
done

echo "== 4. evaluate all on held-out energies =="
$PY -m braggtransporter.evaluate \
    --models experiments/bt/braggtransporter_v0/best.pt \
             experiments/bt/mlp/best.pt \
             experiments/bt/fno1d/best.pt \
             experiments/bt/dota/best.pt \
    --data "$DATA" --out-dir experiments/bt/eval

echo "== 5. compute gate =="
$PY - <<'PYEOF'
import json, pathlib, sys
s = json.loads(pathlib.Path("experiments/bt/eval/summary.json").read_text())
# summary.json is a list of rows; keep aggregate rows, key by model
agg = {}
rows = s if isinstance(s, list) else s.get("summary", [])
import math
def _num(v):
    try:
        f = float(v)
        return 1e9 if math.isnan(f) else f   # nan edge = model can't form a findable Bragg peak = worst
    except Exception:
        return 1e9
for r in rows:
    if str(r.get("energy", r.get("energy_mev", ""))) in ("aggregate", "") or r.get("aggregate"):
        agg[r["model"]] = _num(r["distal_edge_error_mm_mean"])
if not agg:  # fallback: any row per model
    for r in rows:
        agg.setdefault(r["model"], _num(r.get("distal_edge_error_mm_mean", r.get("distal_edge_error_mm", "nan"))))
print("distal_edge_error_mm (held-out mean):")
for k, v in sorted(agg.items(), key=lambda kv: kv[1]):
    print(f"  {k:24s} {v:.3f} mm")
v0 = next((v for k, v in agg.items() if "braggtransporter_v0" in k or k == "braggtransporter_v0"), None)
base = {k: v for k, v in agg.items() if not ("braggtransporter_v0" in k)}
if v0 is None or not base:
    print("GATE: INCONCLUSIVE (missing models)"); sys.exit(2)
worst_needed = max(base.values())
ok = all(v0 <= b + 1e-6 for b in base.values())
print(f"v0={v0:.3f} mm; baselines={ {k: round(b,3) for k,b in base.items()} }")
print("GATE:", "PASS" if ok else "FAIL", "(v0 <= all baselines on distal-edge error)")
sys.exit(0 if ok else 1)
PYEOF