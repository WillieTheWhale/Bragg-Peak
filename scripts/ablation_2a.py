#!/usr/bin/env python
"""Phase 2A representation ablation for BraggTransporter v3.1.

Trains the 2x2x2 matrix:
decoder_mode in {coord_query, fixed_grid}
use_physics_prior in {true, false}
distal_edge_weight in {5.0, 1.0}

Research software only; reports held-out distal-edge error, gamma pass rate, and
RMSE from the existing BraggTransporter evaluation module.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from braggtransporter.config import DataConfig
from braggtransporter.train import SyntheticBraggDataset


DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "bt_v0_1d.yaml"
DEFAULT_DATA = REPO_ROOT / "data" / "generated" / "phase1_1d.h5"
DEFAULT_WORK_DIR = REPO_ROOT / "experiments" / "bt" / "phase2a"
DEFAULT_OUT_CSV = REPO_ROOT / "docs" / "results" / "phase2a.csv"


@dataclass(frozen=True)
class AblationSpec:
    name: str
    decoder_mode: str
    use_physics_prior: bool
    distal_edge_weight: float

    @property
    def loss_mode(self) -> str:
        return "distal_edge_weighted" if self.distal_edge_weight > 1.0 else "uniform"


def ablation_specs() -> list[AblationSpec]:
    specs: list[AblationSpec] = []
    for decoder in ("coord_query", "fixed_grid"):
        decoder_tag = "cq" if decoder == "coord_query" else "fg"
        for prior in (True, False):
            prior_tag = "prior_on" if prior else "prior_off"
            for weight, loss_tag in ((5.0, "weighted"), (1.0, "uniform")):
                specs.append(
                    AblationSpec(
                        name=f"{decoder_tag}_{prior_tag}_{loss_tag}",
                        decoder_mode=decoder,
                        use_physics_prior=prior,
                        distal_edge_weight=weight,
                    )
                )
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs per ablation config.")
    parser.add_argument("--data", type=Path, default=None, help="HDF5 data path. Defaults to phase1_1d.h5.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fast", action="store_true", help="CPU smoke mode with a tiny synthetic HDF5 and one config.")
    parser.add_argument("--max-configs", type=int, default=None, help="Optional smoke/debug limit on matrix size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    data_path = _prepare_data(args, work_dir)

    specs = ablation_specs()
    limit = args.max_configs if args.max_configs is not None else (1 if args.fast else len(specs))
    specs = specs[: max(1, int(limit))]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        cfg_path = _write_config(spec, args, data_path, work_dir)
        ckpt = work_dir / "runs" / spec.name / "braggtransporter_v0" / "best.pt"
        eval_dir = work_dir / "eval" / spec.name
        device_name = "cpu" if args.fast else args.device

        train_cmd = [sys.executable, "-m", "braggtransporter.train", "--config", str(cfg_path), "--device", device_name]
        if args.fast:
            train_cmd.append("--fast")
        _run(train_cmd)

        eval_cmd = [
            sys.executable,
            "-m",
            "braggtransporter.evaluate",
            "--ckpt",
            str(ckpt),
            "--data",
            str(data_path),
            "--out-dir",
            str(eval_dir),
            "--device",
            device_name,
        ]
        _run(eval_cmd)
        rows.append(_row_from_eval(spec, ckpt, eval_dir / "summary.json"))

    ranked = sorted(rows, key=lambda r: _edge_sort_key(r["distal_edge_error_mm"]))
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    _write_phase2a_csv(ranked, args.out_csv)
    _print_ranked_table(ranked)
    for line in _factor_conclusions(ranked):
        print(line)
    print(f"Wrote Phase 2A aggregate CSV to {args.out_csv}")


def _prepare_data(args: argparse.Namespace, work_dir: Path) -> Path:
    if args.fast and args.data is None:
        data_path = work_dir / "tiny_synthetic_phase2a.h5"
        _write_tiny_synthetic_h5(data_path)
        return data_path
    return args.data if args.data is not None else DEFAULT_DATA


def _write_config(spec: AblationSpec, args: argparse.Namespace, data_path: Path, work_dir: Path) -> Path:
    cfg = yaml.safe_load(args.base_config.read_text()) or {}
    cfg.setdefault("data", {})["out_path"] = str(data_path)
    cfg.setdefault("train", {})["model"] = "braggtransporter_v0"
    cfg.setdefault("model", {})
    cfg["train"]["epochs"] = int(args.epochs)
    cfg["train"]["distal_edge_weight"] = float(spec.distal_edge_weight)
    cfg["train"]["out_dir"] = str(work_dir / "runs" / spec.name)
    if args.fast:
        cfg["train"]["device"] = "cpu"
        cfg["train"]["batch_size"] = min(int(cfg["train"].get("batch_size", 8)), 8)
        cfg["model"]["d_model"] = 16
        cfg["model"]["n_layers"] = 1
        cfg["model"]["n_heads"] = 4
        cfg["model"]["d_ff"] = 32
        cfg["model"]["dropout"] = 0.0

    extra = dict(cfg.setdefault("model", {}).get("extra", {}) or {})
    extra["decoder_mode"] = spec.decoder_mode
    extra["use_physics_prior"] = bool(spec.use_physics_prior)
    if args.fast:
        extra["max_positions"] = min(int(extra.get("max_positions", 128)), 128)
    cfg["model"]["extra"] = extra

    cfg_dir = work_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{spec.name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _row_from_eval(spec: AblationSpec, ckpt: Path, summary_json: Path) -> dict[str, Any]:
    payload = json.loads(summary_json.read_text())
    summary = payload.get("summary", payload if isinstance(payload, list) else [])
    aggregate = next((r for r in summary if str(r.get("energy_mev", "")) == "aggregate"), None)
    if aggregate is None:
        aggregate = _mean_summary(summary)
    return {
        "rank": 0,
        "name": spec.name,
        "decoder_mode": spec.decoder_mode,
        "use_physics_prior": spec.use_physics_prior,
        "loss_mode": spec.loss_mode,
        "distal_edge_weight": spec.distal_edge_weight,
        "distal_edge_error_mm": float(aggregate["distal_edge_error_mm_mean"]),
        "gamma_pass_pct": float(aggregate["gamma_pass_pct_mean"]),
        "rmse_pct": float(aggregate["rmse_pct_mean"]),
        "ckpt": str(ckpt),
        "eval_summary_json": str(summary_json),
    }


def _mean_summary(summary: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("distal_edge_error_mm_mean", "gamma_pass_pct_mean", "rmse_pct_mean"):
        vals = [float(row[key]) for row in summary if key in row]
        out[key] = float(np.nanmean(vals)) if vals else float("nan")
    return out


def _write_phase2a_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "name",
        "decoder_mode",
        "use_physics_prior",
        "loss_mode",
        "distal_edge_weight",
        "distal_edge_error_mm",
        "gamma_pass_pct",
        "rmse_pct",
        "ckpt",
        "eval_summary_json",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_ranked_table(rows: list[dict[str, Any]]) -> None:
    print("\nPhase 2A ranked by held-out distal-edge error")
    print(f"{'rank':>4} {'name':>24} {'edge_mm':>9} {'gamma%':>8} {'rmse%':>8}")
    for row in rows:
        print(
            f"{int(row['rank']):>4d} {row['name']:>24} "
            f"{float(row['distal_edge_error_mm']):>9.3f} "
            f"{float(row['gamma_pass_pct']):>8.2f} "
            f"{float(row['rmse_pct']):>8.3f}"
        )


def _factor_conclusions(rows: list[dict[str, Any]]) -> list[str]:
    factors = [
        ("decoder", "decoder_mode", "coord_query", "fixed_grid"),
        ("physics_prior", "use_physics_prior", True, False),
        ("loss", "loss_mode", "distal_edge_weighted", "uniform"),
    ]
    lines: list[str] = []
    for label, key, a, b in factors:
        a_vals = [_edge_sort_key(r["distal_edge_error_mm"]) for r in rows if r[key] == a]
        b_vals = [_edge_sort_key(r["distal_edge_error_mm"]) for r in rows if r[key] == b]
        if not a_vals or not b_vals:
            lines.append(f"{label}: inconclusive; both factor levels were not run.")
            continue
        a_mean = float(np.mean(a_vals))
        b_mean = float(np.mean(b_vals))
        if not np.isfinite(a_mean) and not np.isfinite(b_mean):
            lines.append(f"{label}: inconclusive; both factor levels had undefined distal-edge error.")
            continue
        winner = a if a_mean <= b_mean else b
        delta = abs(a_mean - b_mean)
        lines.append(f"{label}: {winner} reduced held-out distal-edge error by {delta:.3f} mm on average.")
    return lines


def _edge_sort_key(value: Any) -> float:
    val = float(value)
    return val if np.isfinite(val) else float("inf")


def _write_tiny_synthetic_h5(path: Path) -> None:
    import h5py

    cfg = DataConfig(max_depth_cm=12.0, dz_cm=0.25, energies_mev=[90, 120], heldout_energies_mev=[105], seed=7)
    ds = SyntheticBraggDataset(n_samples=12, nz=49, data_cfg=cfg, seed=cfg.seed)
    splits = {
        "train": [ds[i] for i in range(0, 8)],
        "val": [ds[i] for i in range(8, 10)],
        "heldout_energy": [ds[i] for i in range(10, 12)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = "braggtransporter_v3.1_phase2a_tiny_synthetic"
        for name, samples in splits.items():
            group = h5.create_group(name)
            for key in ("x", "scalars", "z", "dose", "letd"):
                group.create_dataset(key, data=np.stack([s[key].numpy() for s in samples]).astype(np.float64))
            group.create_dataset("r80", data=np.asarray([float(s["r80"]) for s in samples], dtype=np.float64))
            group.create_dataset("fidelity", data=np.asarray([float(s["fidelity"]) for s in samples], dtype=np.float64))
            group.create_dataset("energy_mev", data=np.asarray([float(s["scalars"][0]) for s in samples], dtype=np.float64))

        train_x = h5["train/x"][...]
        train_scalars = h5["train/scalars"][...]
        norm = h5.create_group("norm")
        x_mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
        x_std = np.maximum(train_x.reshape(-1, train_x.shape[-1]).std(axis=0), 1.0e-8)
        scalar_mean = train_scalars.mean(axis=0)
        scalar_std = np.maximum(train_scalars.std(axis=0), 1.0e-8)
        scalar_mean[-1] = 0.0
        scalar_std[-1] = 1.0
        norm.create_dataset("x_mean", data=x_mean)
        norm.create_dataset("x_std", data=x_std)
        norm.create_dataset("scalar_mean", data=scalar_mean)
        norm.create_dataset("scalar_std", data=scalar_std)


if __name__ == "__main__":
    main()
