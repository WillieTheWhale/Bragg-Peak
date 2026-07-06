#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ID="${PROJECT_ID:-braggtransporter}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-f}"
BUCKET="${BUCKET:-gs://braggtransporter-braggtransporter}"
INSTANCE="${INSTANCE:-braggtransport-gpu}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-latest-gpu}"
REPO_URL="${REPO_URL:-https://github.com/WillieTheWhale/Bragg-Peak.git}"
SOURCE_GCS="${SOURCE_GCS:-}"
PATIENTS="${PATIENTS:-1ABB006}"
MAX_BEAMLETS="${MAX_BEAMLETS:-256}"
EPOCHS="${EPOCHS:-50}"
MACHINE_KIND="${MACHINE_KIND:-t4}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
RUN_GCS="${RUN_GCS:-}"
TEARDOWN_ONLY=0
TAIL_LOGS=1
CREATED_INSTANCE=0
DELETE_ON_EXIT=1

usage() {
  cat <<'USAGE'
Usage:
  scripts/gcp_train.sh [options]

Creates a cost-safe Spot GPU VM, runs scripts/gcp_bootstrap.sh as startup script,
tails the training logs, and deletes the VM when the training exits.

Options:
  --project ID          GCP project ID (default: braggtransporter)
  --region REGION      GCP region (default: us-central1)
  --zone ZONE          GCP zone (default: us-central1-f)
  --bucket GS_URI      GCS bucket/root (default: gs://braggtransporter-braggtransporter)
  --instance NAME      VM name (default: braggtransport-gpu)
  --patients LIST      Space- or comma-separated DoseRAD patient IDs (default: 1ABB006)
  --max-beamlets N     Beamlet cap passed to training (default: 256)
  --epochs N           Epochs passed to training (default: 50)
  --image-family NAME  Deep Learning VM image family (default: pytorch-latest-gpu)
  --repo-url URL       Git repo cloned on VM (default: Bragg-Peak GitHub repo)
  --source-gcs GS_URI  Optional source tarball copied from GCS instead of git clone
  --run-gcs GS_URI     Exact run output prefix (default: BUCKET/runs/INSTANCE)
  --train-args STRING  Extra arguments appended to train_doserad_gpu.py
  --l4                 Try L4 first, then T4 fallback
  --t4                 Try T4 first, then L4 fallback (default)
  --no-tail            Create and print commands without SSH-tailing logs
  --teardown           Delete the named VM and exit
  -h, --help           Show this help

Environment variables with the same uppercase names may also be used.
USAGE
}

log() {
  printf '[gcp_train] %s\n' "$*"
}

die() {
  printf '[gcp_train] ERROR: %s\n' "$*" >&2
  exit 1
}

normalize_patients() {
  printf '%s' "$1" | tr ',' ' ' | xargs
}

instance_exists() {
  gcloud compute instances describe "$INSTANCE" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --format='value(name)' >/dev/null 2>&1
}

delete_instance() {
  if instance_exists; then
    log "Deleting GPU VM: $INSTANCE"
    printf 'DELETE: gcloud compute instances delete %q --project=%q --zone=%q --quiet\n' "$INSTANCE" "$PROJECT_ID" "$ZONE"
    gcloud compute instances delete "$INSTANCE" \
      --project="$PROJECT_ID" \
      --zone="$ZONE" \
      --quiet || true
  else
    log "No VM named $INSTANCE exists in $ZONE."
  fi
}

on_exit() {
  local status=$?
  if [[ "$DELETE_ON_EXIT" == "1" && "$TEARDOWN_ONLY" == "0" ]]; then
    printf '\n'
    log "EXIT trap active. Cost guard is deleting any remaining GPU VM named $INSTANCE."
    delete_instance
  fi
  exit "$status"
}

trap on_exit EXIT INT TERM

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT_ID="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --instance) INSTANCE="$2"; shift 2 ;;
    --patients) PATIENTS="$2"; shift 2 ;;
    --max-beamlets) MAX_BEAMLETS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --image-family) IMAGE_FAMILY="$2"; shift 2 ;;
    --repo-url) REPO_URL="$2"; shift 2 ;;
    --source-gcs) SOURCE_GCS="$2"; shift 2 ;;
    --run-gcs) RUN_GCS="$2"; shift 2 ;;
    --train-args) TRAIN_ARGS="$2"; shift 2 ;;
    --l4) MACHINE_KIND="l4"; shift ;;
    --t4) MACHINE_KIND="t4"; shift ;;
    --no-tail) TAIL_LOGS=0; shift ;;
    --teardown) TEARDOWN_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

PATIENTS="$(normalize_patients "$PATIENTS")"
RUN_GCS="${RUN_GCS:-${BUCKET%/}/runs/$INSTANCE}"

SSH_CMD=(gcloud compute ssh "$INSTANCE" --project="$PROJECT_ID" --zone="$ZONE")
DELETE_CMD=(gcloud compute instances delete "$INSTANCE" --project="$PROJECT_ID" --zone="$ZONE" --quiet)

printf '\n'
log "Project: $PROJECT_ID"
log "Zone: $ZONE"
log "Bucket/run prefix: $RUN_GCS"
log "LOUD COST REMINDER: GPU VM name is $INSTANCE. Delete it when idle."
printf 'SSH:    '
printf '%q ' "${SSH_CMD[@]}"
printf '\n'
printf 'DELETE: '
printf '%q ' "${DELETE_CMD[@]}"
printf '\n\n'

if [[ "$TEARDOWN_ONLY" == "1" ]]; then
  delete_instance
  exit 0
fi

metadata_arg=(
  "project_id=$PROJECT_ID"
  "region=$REGION"
  "zone=$ZONE"
  "bucket=$BUCKET"
  "run_gcs=$RUN_GCS"
  "repo_url=$REPO_URL"
  "source_gcs=$SOURCE_GCS"
  "patients=$PATIENTS"
  "max_beamlets=$MAX_BEAMLETS"
  "epochs=$EPOCHS"
  "train_args=$TRAIN_ARGS"
)

create_t4() {
  log "Creating Spot T4 VM $INSTANCE in $ZONE."
  gcloud compute instances create "$INSTANCE" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type=n1-standard-8 \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --maintenance-policy=TERMINATE \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-balanced \
    --metadata=install-nvidia-driver=True,"$(IFS=,; printf '%s' "${metadata_arg[*]}")" \
    --metadata-from-file=startup-script=scripts/gcp_bootstrap.sh \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --labels=project=braggtransporter,cost=spot,accelerator=t4
}

create_l4() {
  log "Creating Spot L4 VM $INSTANCE in $ZONE."
  gcloud compute instances create "$INSTANCE" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type=g2-standard-8 \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --maintenance-policy=TERMINATE \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-balanced \
    --metadata=install-nvidia-driver=True,"$(IFS=,; printf '%s' "${metadata_arg[*]}")" \
    --metadata-from-file=startup-script=scripts/gcp_bootstrap.sh \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --labels=project=braggtransporter,cost=spot,accelerator=l4
}

if instance_exists; then
  log "Instance $INSTANCE already exists; reusing it and tailing logs."
else
  if [[ "$MACHINE_KIND" == "l4" ]]; then
    create_l4 || create_t4
  else
    create_t4 || create_l4
  fi
  CREATED_INSTANCE=1
fi

if [[ "$TAIL_LOGS" == "0" ]]; then
  log "--no-tail requested. The EXIT trap will still delete $INSTANCE now."
  exit 0
fi

log "Waiting for SSH and startup logs. This can take several minutes while the DL image boots."
for attempt in $(seq 1 60); do
  if "${SSH_CMD[@]}" --command='true' >/dev/null 2>&1; then
    break
  fi
  if ! instance_exists; then
    die "Instance disappeared before SSH became available."
  fi
  sleep 10
  if [[ "$attempt" == "60" ]]; then
    die "SSH did not become available after 10 minutes."
  fi
done

log "Tailing VM logs until training finishes, fails, or the VM disappears."
set +e
"${SSH_CMD[@]}" --command='sudo bash -lc '"'"'
touch /var/log/braggtransport-bootstrap.log /var/log/braggtransport-training.log
tail -n +1 -F /var/log/braggtransport-bootstrap.log /var/log/braggtransport-training.log &
tail_pid=$!
while true; do
  if [ -f /var/log/braggtransport-training.done ]; then
    sleep 3
    kill "$tail_pid" 2>/dev/null || true
    exit 0
  fi
  if [ -f /var/log/braggtransport-training.failed ]; then
    sleep 3
    kill "$tail_pid" 2>/dev/null || true
    exit 1
  fi
  sleep 20
done
'"'"''
tail_status=$?
set -e

if [[ "$tail_status" -ne 0 ]]; then
  log "SSH tail exited with status $tail_status. Checking whether the Spot VM was preempted or training failed."
fi

if instance_exists; then
  log "Instance still exists after log tail. It will be deleted by the EXIT trap."
else
  log "Instance is already gone."
fi

exit "$tail_status"
