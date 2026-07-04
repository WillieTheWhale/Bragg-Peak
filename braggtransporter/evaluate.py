"""Evaluate BraggTransporter checkpoints on a held-out HDF5 split."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import ModelConfig, get_device
from .metrics import distal_edge_error_mm, gamma_index_1d, peak_depth_cm, r80_r90_r50, rmse_pct


MODEL_REGISTRY = {
    "braggtransporter_v0": "braggtransporter.models.braggtransporter_v0:BraggTransporterV0",
    "v0": "braggtransporter.models.braggtransporter_v0:BraggTransporterV0",
    "mlp": "braggtransporter.models.mlp:MLPBaseline",
    "fno1d": "braggtransporter.models.fno1d:FNO1d",
    "fno": "braggtransporter.models.fno1d:FNO1d",
    "dota": "braggtransporter.models.dota_transformer:DoTATransformer",
    "dota_transformer": "braggtransporter.models.dota_transformer:DoTATransformer",
}


def _import_object(spec: str) -> Any:
    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _checkpoint_model_name(ckpt: dict[str, Any], fallback: str) -> str:
    for key in ("model_name", "model_type", "model"):
        val = ckpt.get(key)
        if isinstance(val, str):
            return val
    cfg = ckpt.get("config")
    if isinstance(cfg, dict) and isinstance(cfg.get("model"), str):
        return str(cfg["model"])
    # train.py stores the model name at config["train"]["model"] (config["model"]
    # is the ModelConfig dict).
    if isinstance(cfg, dict) and isinstance(cfg.get("train"), dict) and isinstance(cfg["train"].get("model"), str):
        return str(cfg["train"]["model"])
    train_cfg = ckpt.get("train_config")
    if isinstance(train_cfg, dict) and isinstance(train_cfg.get("model"), str):
        return str(train_cfg["model"])
    return fallback


def _model_kwargs(ckpt: dict[str, Any]) -> dict[str, Any]:
    for key in ("model_config", "model_kwargs"):
        val = ckpt.get(key)
        if isinstance(val, dict):
            return dict(val)
    cfg = ckpt.get("config")
    if isinstance(cfg, dict) and isinstance(cfg.get("model_config"), dict):
        return dict(cfg["model_config"])
    return asdict(ModelConfig())


def _state_dict(ckpt: Any) -> dict[str, Any] | None:
    if not isinstance(ckpt, dict):
        return None
    for key in ("model_state_dict", "state_dict", "model_state"):
        val = ckpt.get(key)
        if isinstance(val, dict):
            return val
    val = ckpt.get("model")
    if isinstance(val, dict):
        return val
    return None


def load_model(ckpt_path: Path, device_name: str = "auto") -> Any:
    """Load a checkpoint using the Module B/C model registry."""
    import torch

    device = get_device(device_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    if hasattr(ckpt, "eval") and callable(ckpt):
        model = ckpt
    elif isinstance(ckpt, dict) and hasattr(ckpt.get("model"), "eval"):
        model = ckpt["model"]
    elif isinstance(ckpt, dict):
        model_name = _checkpoint_model_name(ckpt, ckpt_path.stem)
        spec = MODEL_REGISTRY.get(model_name, model_name if ":" in model_name else "")
        if not spec:
            raise ValueError(f"Cannot infer model class for checkpoint {ckpt_path}.")
        cls = _import_object(spec)
        kwargs = _model_kwargs(ckpt)
        try:
            model = cls(**kwargs)
        except TypeError:
            model = cls(ModelConfig(**{k: v for k, v in kwargs.items() if hasattr(ModelConfig, k)}))
        state = _state_dict(ckpt)
        if state is None:
            raise ValueError(f"Checkpoint {ckpt_path} does not contain a model state_dict.")
        model.load_state_dict(state)
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    model.to(device)
    model.eval()
    return model


def _choose_group(h5: Any) -> Any:
    for name in ("heldout", "heldout-energy", "heldout_energy", "test", "val", "validation"):
        if name in h5:
            return h5[name]
    return h5


class H5HeldoutDataset:
    """Minimal HDF5 dataset reader for the fixed BraggTransporter tensor contract."""

    def __init__(self, path: Path):
        import h5py

        self.h5 = h5py.File(path, "r")
        self.group = _choose_group(self.h5)
        required = ("x", "scalars", "z", "dose")
        missing = [name for name in required if name not in self.group]
        if missing:
            raise ValueError(f"HDF5 held-out split is missing datasets: {missing}")
        self.n = int(self.group["dose"].shape[0])
        # Apply the SAME train-split standardization BraggDataset applies, or the
        # model (trained on standardized inputs) sees raw inputs here and predicts
        # garbage. Targets (dose/z) stay in physical units.
        if "norm" in self.h5:
            self._x_mean = np.asarray(self.h5["norm/x_mean"][...], dtype=np.float64)
            self._x_std = np.asarray(self.h5["norm/x_std"][...], dtype=np.float64)
            self._scalar_mean = np.asarray(self.h5["norm/scalar_mean"][...], dtype=np.float64)
            self._scalar_std = np.asarray(self.h5["norm/scalar_std"][...], dtype=np.float64)
        else:
            self._x_mean = self._scalar_mean = 0.0
            self._x_std = self._scalar_std = 1.0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {name: np.asarray(self.group[name][idx]) for name in ("x", "scalars", "z", "dose")}
        item["x"] = (item["x"] - self._x_mean) / self._x_std
        item["scalars"] = (item["scalars"] - self._scalar_mean) / self._scalar_std
        if "energy_mev" in self.group:
            item["energy_mev"] = float(np.asarray(self.group["energy_mev"][idx]))
        else:
            item["energy_mev"] = float(item["scalars"][0])
        return item

    def close(self) -> None:
        self.h5.close()


def _batch_iter(dataset: H5HeldoutDataset, batch_size: int) -> Iterable[dict[str, np.ndarray]]:
    for start in range(0, len(dataset), batch_size):
        rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        yield {
            "x": np.stack([r["x"] for r in rows]),
            "scalars": np.stack([r["scalars"] for r in rows]),
            "z": np.stack([r["z"] for r in rows]),
            "dose": np.stack([r["dose"] for r in rows]),
            "energy_mev": np.asarray([r["energy_mev"] for r in rows], dtype=np.float64),
        }


def _predict(model: Any, batch: dict[str, np.ndarray], device_name: str) -> np.ndarray:
    import torch

    device = get_device(device_name)
    x = torch.as_tensor(batch["x"], dtype=torch.float32, device=device)
    scalars = torch.as_tensor(batch["scalars"], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(x, scalars)
    dose = out["dose"] if isinstance(out, dict) else out
    return dose.detach().cpu().numpy()


def _sample_metrics(model_name: str, energy: float, z: np.ndarray, pred: np.ndarray, ref: np.ndarray) -> dict[str, float | str]:
    ranges = r80_r90_r50(z, pred)
    ref_ranges = r80_r90_r50(z, ref)
    return {
        "model": model_name,
        "energy_mev": float(energy),
        "rmse_pct": rmse_pct(pred, ref),
        "gamma_pass_pct": gamma_index_1d(pred, ref, z),
        "distal_edge_error_mm": distal_edge_error_mm(pred, ref, z),
        "peak_depth_cm": peak_depth_cm(z, pred),
        "r80_mm": ranges["r80_mm"],
        "r80_ref_mm": ref_ranges["r80_mm"],
        "r90_mm": ranges["r90_mm"],
        "r50_mm": ranges["r50_mm"],
    }


def evaluate_checkpoint(ckpt_path: Path, data_path: Path, batch_size: int, device_name: str) -> list[dict[str, float | str]]:
    import torch

    model = load_model(ckpt_path, device_name)
    # Label by the inferred model name (all checkpoints are named best.pt in
    # per-model dirs, so the filename stem is not unique). Fall back to the parent
    # directory name, which is unique per model in the gate layout.
    _raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    label = _checkpoint_model_name(_raw, ckpt_path.parent.name) if isinstance(_raw, dict) else ckpt_path.parent.name
    dataset = H5HeldoutDataset(data_path)
    rows: list[dict[str, float | str]] = []
    try:
        for batch in _batch_iter(dataset, batch_size):
            pred = _predict(model, batch, device_name)
            for i in range(pred.shape[0]):
                rows.append(
                    _sample_metrics(
                        label,
                        float(batch["energy_mev"][i]),
                        np.asarray(batch["z"][i], dtype=np.float64),
                        np.asarray(pred[i], dtype=np.float64),
                        np.asarray(batch["dose"][i], dtype=np.float64),
                    )
                )
    finally:
        dataset.close()
    return rows


def summarise(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    grouped: dict[tuple[str, str], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), f"{float(row['energy_mev']):.6g}")].append(row)
        grouped[(str(row["model"]), "aggregate")].append(row)

    summary: list[dict[str, float | str]] = []
    metric_names = ("rmse_pct", "gamma_pass_pct", "distal_edge_error_mm", "peak_depth_cm")
    for (model, energy), vals in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1] == "aggregate", kv[0][1])):
        out: dict[str, float | str] = {"model": model, "energy_mev": energy, "n": float(len(vals))}
        for name in metric_names:
            arr = np.asarray([float(v[name]) for v in vals], dtype=np.float64)
            out[f"{name}_mean"] = float(np.nanmean(arr))
            out[f"{name}_std"] = float(np.nanstd(arr))
        summary.append(out)
    return summary


def print_table(summary: list[dict[str, float | str]]) -> None:
    headers = ["model", "energy", "n", "rmse%", "gamma%", "edge_mm", "peak_cm"]
    print(" ".join(f"{h:>12}" for h in headers))
    for row in summary:
        print(
            f"{str(row['model']):>12} {str(row['energy_mev']):>12} {int(float(row['n'])):>12d} "
            f"{float(row['rmse_pct_mean']):>12.3f} {float(row['gamma_pass_pct_mean']):>12.2f} "
            f"{float(row['distal_edge_error_mm_mean']):>12.3f} {float(row['peak_depth_cm_mean']):>12.3f}"
        )


def write_outputs(rows: list[dict[str, float | str]], summary: list[dict[str, float | str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = out_dir / "samples.csv"
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"

    with sample_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    summary_json.write_text(json.dumps({"samples_csv": str(sample_csv), "summary": summary}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, help="Single checkpoint path.")
    parser.add_argument("--models", nargs="+", type=Path, help="Checkpoint paths to compare side by side.")
    parser.add_argument("--data", type=Path, required=True, help="Held-out HDF5 dataset.")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/bt/eval"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpts = args.models if args.models else ([args.ckpt] if args.ckpt else [])
    if not ckpts:
        raise SystemExit("Provide --ckpt <pt> or --models a.pt b.pt ...")

    rows: list[dict[str, float | str]] = []
    for ckpt in ckpts:
        rows.extend(evaluate_checkpoint(ckpt, args.data, args.batch_size, args.device))
    if not rows:
        raise SystemExit("No evaluation samples were produced.")

    summary = summarise(rows)
    print_table(summary)
    write_outputs(rows, summary, args.out_dir)
    print(f"Wrote CSV and JSON evaluation artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
