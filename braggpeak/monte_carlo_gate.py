"""OpenGATE/Geant4 high-fidelity reference adapter.

Wraps a GATE 10 (OpenGATE) proton depth-dose simulation behind the same
:class:`~braggpeak.transport.DepthDose` interface the candidate models use, so a
Geant4 Monte Carlo reference drops into the validation harness unchanged. The
``opengate`` package is an optional dependency: importing this module never
fails, but calling the simulator without it raises a clear, actionable error.

Reference outputs record the physics list, seed, primary count, voxel size and
GATE/Geant4 versions in ``metadata`` so they never collide with candidate
outputs. Coordinate frame: beam along +z, phantom entrance at z=0.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from .materials import Material, WATER, get_material
from .transport import DepthDose, Slab


# Map braggpeak reference materials to Geant4/NIST material names.
_G4_MATERIAL = {
    "liquid_water": "G4_WATER",
    "icrp_soft_tissue": "G4_TISSUE_SOFT_ICRP",
    "icru_cortical_bone": "G4_BONE_COMPACT_ICRU",
    "icru_lung_inflated": "G4_LUNG_ICRP",
    "aluminium": "G4_Al",
    "air": "G4_AIR",
}


def opengate_available() -> bool:
    """True if the optional ``opengate`` + Geant4 core packages are installed.

    Uses import-spec discovery rather than importing the heavy Geant4 binding,
    so merely checking availability never loads Geant4 into the current process
    (all actual simulations run in an isolated subprocess).
    """
    import importlib.util

    return (
        importlib.util.find_spec("opengate") is not None
        and importlib.util.find_spec("opengate_core") is not None
    )


def g4_material_name(material: Material) -> str:
    """Geant4 material name for a braggpeak material (raises if unmapped)."""
    try:
        return _G4_MATERIAL[material.name]
    except KeyError as exc:
        raise KeyError(f"No Geant4 material mapping for '{material.name}'.") from exc


def simulate_depth_dose_gate(
    energy_mev: float,
    slabs: Sequence[Slab],
    *,
    dz_cm: float = 0.05,
    lateral_cm: float = 6.0,
    n_primaries: int = 20000,
    seed: int = 1234,
    energy_spread_pct: float = 0.8,
    physics_list: str = "QGSP_BIC_EMZ",
    n_threads: int = 1,
    output_dir: str | None = None,
) -> DepthDose:
    """Run a GATE 10 proton depth-dose reference in an isolated subprocess.

    Geant4 cannot initialise more than one simulation per Python process, so
    each call runs the actual simulation in a fresh subprocess (via the module
    ``__main__`` worker below) and reads back the depth profile. This keeps the
    energy-ladder loop and pytest safe. Returns a :class:`DepthDose` with the
    integral depth-dose and full physics provenance.
    """
    if not opengate_available():
        raise RuntimeError(
            "OpenGATE reference requires the optional dependency. Install with "
            "`pip install 'braggpeak[gate]'` (pulls a Geant4-backed opengate_core)."
        )
    work = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="gate_"))
    work.mkdir(parents=True, exist_ok=True)
    args = {
        "energy_mev": energy_mev,
        "slabs": [{"material": s.material.name, "thickness_cm": s.thickness_cm} for s in slabs],
        "dz_cm": dz_cm, "lateral_cm": lateral_cm, "n_primaries": n_primaries,
        "seed": seed, "energy_spread_pct": energy_spread_pct,
        "physics_list": physics_list, "n_threads": n_threads, "work": str(work),
    }
    args_path = work / "args.json"
    args_path.write_text(json.dumps(args))
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; from braggpeak.monte_carlo_gate import _worker_main;"
         " sys.exit(_worker_main(sys.argv[1]))", str(args_path)],
        capture_output=True, text=True,
    )
    result_path = work / "result.npz"
    if proc.returncode != 0 or not result_path.exists():
        raise RuntimeError(
            f"GATE subprocess failed (code {proc.returncode}).\n{proc.stderr[-2000:]}"
        )
    data = np.load(result_path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"]))
    n = data["dose"].shape[0]
    return DepthDose(
        z_cm=data["z_cm"], dose=data["dose"], dose_ideal=data["dose"].copy(),
        energy_mev=np.zeros(n), lin_stopping=np.zeros(n), let_kev_um=np.zeros(n),
        metadata=metadata,
    )


def _simulate_in_process(
    energy_mev: float,
    slabs: Sequence[Slab],
    *,
    dz_cm: float,
    lateral_cm: float,
    n_primaries: int,
    seed: int,
    energy_spread_pct: float,
    physics_list: str,
    n_threads: int,
    output_dir: str,
) -> DepthDose:
    """The actual GATE simulation body. Runs at most once per process."""
    import opengate as gate

    u = gate.g4_units
    total_depth_cm = float(sum(s.thickness_cm for s in slabs))
    n_vox = int(round(total_depth_cm / dz_cm))

    sim = gate.Simulation()
    sim.random_seed = seed
    sim.number_of_threads = n_threads
    sim.progress_bar = False
    sim.g4_verbose = False
    sim.visu = False

    # World spans well beyond the phantom (entrance at z=0, stack along +z, plus
    # a 5 cm upstream gap for the source) so no volume protrudes.
    sim.world.size = [lateral_cm * 2 * u.cm + 4 * u.cm,
                      lateral_cm * 2 * u.cm + 4 * u.cm,
                      (total_depth_cm * 2 + 40.0) * u.cm]
    sim.world.material = "G4_AIR"

    # Use standard Geant4 NIST materials: their ICRU mean-excitation energies
    # match the braggpeak reference materials exactly (water 78, bone 91.9,
    # soft-tissue 72.3, air 85.7 eV), so candidate and reference agree on
    # stopping power. (add_material_weights cannot set I, which would bias range.)
    # Stack slabs along z; entrance face of the first slab at world z origin.
    z_cursor = 0.0
    phantom_names = []
    for i, s in enumerate(slabs):
        vol = sim.add_volume("Box", f"slab_{i}")
        vol.size = [lateral_cm * 2 * u.cm, lateral_cm * 2 * u.cm, s.thickness_cm * u.cm]
        center_z = z_cursor + s.thickness_cm / 2.0
        vol.translation = [0, 0, center_z * u.cm]
        vol.material = g4_material_name(s.material)
        phantom_names.append(vol.name)
        z_cursor += s.thickness_cm

    source = sim.add_source("GenericSource", "beam")
    source.particle = "proton"
    source.energy.mono = energy_mev * u.MeV
    if energy_spread_pct > 0:
        source.energy.type = "gauss"
        source.energy.mono = energy_mev * u.MeV
        source.energy.sigma_gauss = energy_spread_pct / 100.0 * energy_mev * u.MeV
    source.position.type = "disc"
    source.position.radius = 2.0 * u.mm
    source.position.translation = [0, 0, -1.0 * u.cm]
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.n = n_primaries

    dose = sim.add_actor("DoseActor", "depth_dose")
    dose.attached_to = "world"
    dose.size = [1, 1, n_vox]
    dose.spacing = [lateral_cm * 2 * u.cm, lateral_cm * 2 * u.cm, dz_cm * u.cm]
    dose.translation = [0, 0, total_depth_cm / 2.0 * u.cm]
    tmp = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="gate_"))
    tmp.mkdir(parents=True, exist_ok=True)
    dose.output_filename = str(tmp / "dose.mhd")
    dose.dose.active = True
    dose.edep.active = True

    sim.physics_manager.physics_list_name = physics_list
    sim.run()

    edep = _read_depth_profile(dose)
    z_cm = (np.arange(n_vox) + 0.5) * dz_cm
    # DoseActor edep is total energy per voxel (MeV); convert to MeV/g per primary.
    densities = _voxel_densities(slabs, z_cm)
    mass_per_voxel_g = densities * (lateral_cm * 2) ** 2 * dz_cm  # g (cm^3 * g/cm^3)
    dose_mev_g = edep / np.maximum(mass_per_voxel_g, 1e-12) / n_primaries

    metadata = {
        "model": "opengate_geant4",
        "initial_energy_mev": energy_mev,
        "dz_cm": dz_cm,
        "n_primaries": n_primaries,
        "energy_spread_pct": energy_spread_pct,
        "seed": seed,
        "physics_list": physics_list,
        "n_threads": n_threads,
        "gate_version": getattr(gate, "__version__", "unknown"),
        "slabs": [
            {"material": s.material.name, "thickness_cm": s.thickness_cm,
             "g4_material": g4_material_name(s.material),
             "density_g_cm3": s.material.density_g_cm3}
            for s in slabs
        ],
        "units": {"z": "cm", "dose": "MeV/g", "energy": "MeV"},
    }
    return DepthDose(
        z_cm=z_cm, dose=dose_mev_g, dose_ideal=dose_mev_g.copy(),
        energy_mev=np.zeros(n_vox), lin_stopping=np.zeros(n_vox),
        let_kev_um=np.zeros(n_vox), metadata=metadata,
    )


def _read_depth_profile(dose_actor) -> np.ndarray:
    """Extract the 1-D edep profile (length n_vox) from the dose actor output."""
    import itk

    path = str(dose_actor.edep.get_output_path())
    arr = itk.array_from_image(itk.imread(path))  # shape (nz, ny, nx)
    return np.asarray(arr, dtype=np.float64).sum(axis=(1, 2))


def _voxel_densities(slabs: Sequence[Slab], z_cm: np.ndarray) -> np.ndarray:
    """Per-voxel mass density (g/cm^3) matching the depth grid."""
    boundaries = np.cumsum([s.thickness_cm for s in slabs])
    out = np.empty_like(z_cm)
    for i, zc in enumerate(z_cm):
        idx = int(np.searchsorted(boundaries, zc, side="right"))
        idx = min(idx, len(slabs) - 1)
        out[i] = slabs[idx].material.density_g_cm3
    return out


def _worker_main(args_path: str) -> int:
    """Subprocess entry point: run one GATE simulation and save result.npz."""
    args = json.loads(Path(args_path).read_text())
    slabs = [Slab(get_material(s["material"]), float(s["thickness_cm"])) for s in args["slabs"]]
    result = _simulate_in_process(
        args["energy_mev"], slabs,
        dz_cm=args["dz_cm"], lateral_cm=args["lateral_cm"],
        n_primaries=args["n_primaries"], seed=args["seed"],
        energy_spread_pct=args["energy_spread_pct"],
        physics_list=args["physics_list"], n_threads=args["n_threads"],
        output_dir=args["work"],
    )
    np.savez(
        Path(args["work"]) / "result.npz",
        z_cm=result.z_cm, dose=result.dose,
        metadata=json.dumps(result.metadata),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main(sys.argv[1]))
