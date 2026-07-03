"""Command-line entry point for braggpeak.

Subcommands
-----------
``run``      Simulate one depth-dose curve from a YAML config; write artifacts.
``metrics``  Print Bragg metrics for a saved CSV depth-dose curve.
``version``  Print the package version.

Configs are the single source of truth for a run: every physics input is read
from YAML and echoed into the output metadata, so results regenerate from one
command with pinned inputs and a fixed seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from . import __version__
from .materials import get_material
from .transport import Slab, simulate_depth_dose
from .scoring import compute_bragg_metrics
from .io import save_depth_dose


def _load_config(path: str | Path) -> dict:
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a mapping.")
    return cfg


def _slabs_from_config(cfg: dict) -> list[Slab]:
    slabs_cfg = cfg.get("slabs")
    if not slabs_cfg:
        raise ValueError("Config must define a non-empty 'slabs' list.")
    slabs = []
    for entry in slabs_cfg:
        material = get_material(entry["material"])
        slabs.append(Slab(material, float(entry["thickness_cm"])))
    return slabs


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    slabs = _slabs_from_config(cfg)
    beam = cfg.get("beam", {})
    numerics = cfg.get("numerics", {})

    result = simulate_depth_dose(
        energy_mev=float(beam["energy_mev"]),
        slabs=slabs,
        dz_cm=float(numerics.get("dz_cm", 0.01)),
        energy_spread_pct=float(beam.get("energy_spread_pct", 0.8)),
        e_cut_mev=float(numerics.get("e_cut_mev", 1.0)),
        fluence0=float(beam.get("fluence0", 1.0)),
        seed=int(cfg.get("seed", 0)),
    )
    metrics = compute_bragg_metrics(result.z_cm, result.dose).as_dict()

    stem = cfg.get("name", Path(args.config).stem)
    out_dir = args.out or cfg.get("output_dir", "experiments/output")
    paths = save_depth_dose(result, out_dir, stem, metrics=metrics, make_png=not args.no_png)

    print(f"Run '{stem}' complete. Artifacts:")
    for kind, p in paths.items():
        print(f"  {kind:5s} {p}")
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    data = np.loadtxt(args.csv, delimiter=",", skiprows=1)
    z_cm, dose = data[:, 0], data[:, 1]
    metrics = compute_bragg_metrics(z_cm, dose).as_dict()
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_version(_args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="braggpeak", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Simulate a depth-dose curve from a YAML config.")
    p_run.add_argument("config", help="Path to a run config (YAML).")
    p_run.add_argument("--out", help="Output directory (overrides config).")
    p_run.add_argument("--no-png", action="store_true", help="Skip PNG plotting.")
    p_run.set_defaults(func=cmd_run)

    p_metrics = sub.add_parser("metrics", help="Print metrics for a saved depth-dose CSV.")
    p_metrics.add_argument("csv", help="Depth-dose CSV (depth_cm,dose,...).")
    p_metrics.set_defaults(func=cmd_metrics)

    p_ver = sub.add_parser("version", help="Print version.")
    p_ver.set_defaults(func=cmd_version)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
