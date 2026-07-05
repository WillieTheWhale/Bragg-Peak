#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
mkdir -p experiments/bt/2b
gen_cfg(){ $PY - "$1" "$2" "$3" <<PYEOF
import yaml,pathlib,sys
m,ep,out=sys.argv[1],int(sys.argv[2]),sys.argv[3]
b=yaml.safe_load(pathlib.Path("configs/bt_v0_1d.yaml").read_text())
b["train"]["model"]=m; b["train"]["epochs"]=ep; b["train"]["device"]="auto"
b["train"]["out_dir"]="experiments/bt/2b"
pathlib.Path(out).write_text(yaml.safe_dump(b))
PYEOF
}
# v0 and fno: matched 40 epochs; mamba: bounded 12 (intractable to match)
gen_cfg braggtransporter_v0 40 configs/_2b_v0.yaml
gen_cfg fno1d 40 configs/_2b_fno.yaml
gen_cfg mamba1d 12 configs/_2b_mamba.yaml
for c in v0 fno mamba; do
  echo "=== training $c ==="; PYTORCH_ENABLE_MPS_FALLBACK=0 $PY -m braggtransporter.train --config configs/_2b_$c.yaml
done
echo "=== evaluate all three on held-out ==="
PYTORCH_ENABLE_MPS_FALLBACK=0 $PY -m braggtransporter.evaluate \
  --models experiments/bt/2b/braggtransporter_v0/best.pt experiments/bt/2b/fno1d/best.pt experiments/bt/2b/mamba1d/best.pt \
  --data data/generated/phase1_1d.h5 --out-dir experiments/bt/2b/eval --device cpu 2>&1 | grep -vE "UserWarning|resize_|x_ft|irfft|Triggered|run_backward"
echo "=== DONE 2B focused ==="
