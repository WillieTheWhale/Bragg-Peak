#!/bin/bash
# Autonomous DoseRAD2026 3-D scaling run. Runs as VM startup script (root).
# Downloads a large beamlet set, trains a scaled Bragg3D, streams progress to GCS,
# then SELF-DELETES the VM. No SSH babysitting; cannot orphan a costly VM.
set -x
LOG=/var/log/bt_scaling.log
exec > >(tee -a "$LOG") 2>&1
BUCKET="gs://braggtransporter-braggtransporter"
RUN="$BUCKET/runs/scaling2"
PATIENTS="1ABB006,1ABB011,1ABB020,1ABB021,1ABB030,1ABB031,1ABB035,1ABB036,1ABB039,1ABB041,1ABB042,1ABB045,1ABB061,1ABB067,1ABB070,1ABB078,1ABB083,1ABB098,1ABB102,1ABB109,1ABB111,1ABB114,1ABB120,1ABB123,1ABB130,1ABB131,1ABB140,1ABB141,1ABB150,1ABB151"
PER_PATIENT=200

push() { gsutil -q cp "$LOG" "$RUN/startup.log" 2>/dev/null || true; }
finish() {
  push
  gsutil -q cp /opt/bt/train.log "$RUN/train.log" 2>/dev/null || true
  echo "DONE $(date -u +%FT%TZ)" | gsutil -q cp - "$RUN/DONE" 2>/dev/null || true
  # self-delete
  NAME=$(hostname)
  ZONE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
  gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet || true
}
trap finish EXIT

apt-get update -y && apt-get install -y --no-install-recommends git python3-venv python3-pip ca-certificates jq
rm -rf /opt/bt && git clone --depth 1 https://github.com/WillieTheWhale/Bragg-Peak.git /opt/bt
cd /opt/bt
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -q --upgrade pip wheel setuptools
.venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || .venv/bin/python -m pip install -q torch
.venv/bin/python -m pip install -q SimpleITK huggingface_hub numpy scipy h5py pyyaml einops google-cloud-storage
push

# background progress pusher (every 3 min)
( while true; do sleep 180; push; done ) &

echo "=== DOWNLOAD $(date -u +%FT%TZ) ==="
.venv/bin/python -u scripts/vm_download_doserad.py "$PATIENTS" "$PER_PATIENT" || true
NBEAM=$(find data/doserad2026 -path '*/dose/*.mha' -size +50000c | wc -l)
echo "VALID_BEAMLETS=$NBEAM"
push

echo "=== TRAIN (scaled Bragg3D) $(date -u +%FT%TZ) ==="
PATS=$(ls data/doserad2026 | grep 1ABB | tr '\n' ' ')
.venv/bin/python -u scripts/train_doserad_gpu.py --patients $PATS \
  --max-beamlets "$PER_PATIENT" --epochs 120 --device cuda --batch-size 8 \
  --d-model 192 --n-layers 6 --lr 3e-4 \
  --gcs "$RUN" --out-dir /opt/bt/runs/scaling2 2>&1 | tee /opt/bt/train.log
echo "=== TRAIN DONE $(date -u +%FT%TZ) ==="
# finish() runs via EXIT trap: uploads logs, writes DONE, self-deletes
