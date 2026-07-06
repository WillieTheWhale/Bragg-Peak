# GCP Spot GPU Runbook

Research software only. These commands launch one cost-guarded Spot GPU VM for
BraggTransporter DoseRAD training and delete it when the launcher exits.

## Defaults

- Project: `braggtransporter`
- Region: `us-central1`
- Zone: `us-central1-f`
- Bucket: `gs://braggtransporter-braggtransporter`
- Default instance: `braggtransport-gpu`
- Default accelerator: 1x NVIDIA T4 Spot, with L4 fallback
- Image project: `deeplearning-platform-release`
- Image family: `pytorch-latest-gpu`
- Boot disk: 100 GB balanced persistent disk

## Launch

Review the scripts first. The launcher creates the VM, tails the bootstrap and
training logs over SSH, and deletes the VM on completion or local exit.

```bash
scripts/gcp_train.sh \
  --patients "1ABB006" \
  --max-beamlets 256 \
  --epochs 50
```

Use an explicit run prefix when needed:

```bash
scripts/gcp_train.sh \
  --instance braggtransport-gpu-o2 \
  --run-gcs gs://braggtransporter-braggtransporter/runs/o2-doserad \
  --patients "1ABB006" \
  --max-beamlets 512 \
  --epochs 100
```

Try L4 first instead of T4:

```bash
scripts/gcp_train.sh --l4 --patients "1ABB006" --max-beamlets 512 --epochs 100
```

## Watch

In another terminal, poll the VM state, latest GCS checkpoint or metrics files,
and billing-account availability. Current same-day spend is not exposed by
`gcloud billing` unless billing export is configured, so the watcher reports
`spend=n/a` when exact spend is unavailable.

```bash
.venv/bin/python scripts/gcp_watch.py \
  --instance braggtransport-gpu \
  --run-gcs gs://braggtransporter-braggtransporter/runs/braggtransport-gpu \
  --max-hours 10
```

If the watcher prints `WARN=max-hours-exceeded(10h)`, delete the VM unless an
active training run is intentionally still using it.

## SSH

The launcher prints this command every run:

```bash
gcloud compute ssh braggtransport-gpu \
  --project=braggtransporter \
  --zone=us-central1-f
```

Useful VM-side checks:

```bash
sudo tail -n 200 -f /var/log/braggtransport-bootstrap.log
sudo tail -n 200 -f /var/log/braggtransport-training.log
nvidia-smi
```

## Fetch Results

Training writes checkpoints, logs, markers, and copied artifacts under the run
prefix in GCS.

```bash
mkdir -p experiments/gcp_fetch/braggtransport-gpu
gcloud storage cp --recursive \
  gs://braggtransporter-braggtransporter/runs/braggtransport-gpu \
  experiments/gcp_fetch/
```

Check completion markers:

```bash
gcloud storage ls gs://braggtransporter-braggtransporter/runs/braggtransport-gpu/
```

## Delete

Always delete idle GPU VMs. Spot T4 is roughly `$0.11/hr`, but idle GPUs still
burn budget and quota.

```bash
gcloud compute instances delete braggtransport-gpu \
  --project=braggtransporter \
  --zone=us-central1-f \
  --quiet
```

Or use the harness cleanup path:

```bash
scripts/gcp_train.sh --instance braggtransport-gpu --teardown
```
