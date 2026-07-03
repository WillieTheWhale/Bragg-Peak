"""Output artifacts for depth-dose runs: CSV, NPZ, PNG, and metadata JSON.

Every run writes numeric data (CSV + NPZ) plus a metadata JSON so that any plot
can be regenerated from saved numbers -- no figure is the sole record of a
result. PNG plotting is optional and guarded behind matplotlib so the core
package has no hard plotting dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .transport import DepthDose


def save_depth_dose(
    result: DepthDose,
    out_dir: str | Path,
    stem: str,
    *,
    metrics: Optional[dict] = None,
    make_png: bool = True,
) -> dict[str, Path]:
    """Write CSV, NPZ, metadata JSON, and (optionally) PNG for a run.

    Returns a mapping of artifact kind -> written path. Directories are created
    as needed. The CSV header names every column with its unit.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    csv_path = out / f"{stem}.csv"
    header = "depth_cm,dose_MeV_per_g,dose_ideal_MeV_per_g,energy_MeV,lin_stopping_MeV_per_cm,let_keV_per_um"
    data = np.column_stack(
        [result.z_cm, result.dose, result.dose_ideal, result.energy_mev,
         result.lin_stopping, result.let_kev_um]
    )
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    paths["csv"] = csv_path

    npz_path = out / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        z_cm=result.z_cm,
        dose=result.dose,
        dose_ideal=result.dose_ideal,
        energy_mev=result.energy_mev,
        lin_stopping=result.lin_stopping,
        let_kev_um=result.let_kev_um,
    )
    paths["npz"] = npz_path

    meta = dict(result.metadata)
    if metrics is not None:
        meta["metrics"] = metrics
    meta_path = out / f"{stem}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    paths["meta"] = meta_path

    if make_png:
        png_path = _try_plot(result, out / f"{stem}.png")
        if png_path is not None:
            paths["png"] = png_path

    return paths


def _try_plot(result: DepthDose, png_path: Path) -> Optional[Path]:
    """Plot depth-dose + energy if matplotlib is available; else skip."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(result.z_cm, result.normalised(), label="broadened (normalised)")
    peak = result.dose_ideal.max()
    if peak > 0:
        ax1.plot(result.z_cm, result.dose_ideal / peak, alpha=0.5, label="ideal CSDA")
    ax1.set_ylabel("relative dose [a.u.]")
    ax1.set_title(
        f"Bragg curve: {result.metadata.get('initial_energy_mev')} MeV, "
        f"{result.metadata['slabs'][0]['material']}"
    )
    ax1.legend()
    ax2.plot(result.z_cm, result.energy_mev, color="C3")
    ax2.set_xlabel("depth [cm]")
    ax2.set_ylabel("proton energy [MeV]")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return png_path
