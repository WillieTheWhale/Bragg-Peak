"""OpenGATE/Geant4 high-fidelity reference adapter (optional dependency).

This wraps an OpenGATE (GATE 10) simulation behind the same depth-dose
interface the candidate models use, so a Geant4 Monte Carlo reference can be
dropped into the validation harness without changing callers. It is guarded:
importing this module never fails, but calling :func:`simulate_depth_dose_gate`
without the ``opengate`` extra installed raises a clear, actionable error.

Install the reference engine with ``pip install 'braggpeak[gate]'`` and pin the
Geant4/GATE data versions in benchmark metadata (physics list, cuts, seed).
"""

from __future__ import annotations

from typing import Sequence

from .transport import Slab


def opengate_available() -> bool:
    """True if the optional ``opengate`` package can be imported."""
    try:
        import opengate  # noqa: F401
    except Exception:
        return False
    return True


def simulate_depth_dose_gate(
    energy_mev: float,
    slabs: Sequence[Slab],
    *,
    dz_cm: float = 0.02,
    n_primaries: int = 100_000,
    seed: int = 0,
    physics_list: str = "QGSP_BIC_EMZ",
):
    """Run an OpenGATE proton depth-dose simulation (reference branch).

    Not yet implemented against a built Geant4; this is the integration seam.
    The intended contract: build a water/slab geometry from ``slabs``, a
    monoenergetic proton pencil-beam source at ``energy_mev``, a depth-dose
    actor with voxel size ``dz_cm``, run ``n_primaries`` histories with the
    named ``physics_list`` and ``seed``, and return a
    :class:`~braggpeak.transport.DepthDose` with full physics-list/version
    provenance in ``metadata`` so it never collides with candidate outputs.
    """
    if not opengate_available():
        raise RuntimeError(
            "OpenGATE reference requires the optional dependency. Install with "
            "`pip install 'braggpeak[gate]'` and a built Geant4/GATE 10 backend."
        )
    raise NotImplementedError(
        "OpenGATE adapter is a planned integration seam; the geometry/source/"
        "scorer wiring lands once a Geant4 backend is available in CI."
    )
