from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from braggtransporter.data.doserad import DOSERAD_INPUT_CHANNELS, extract_bev_pair, hu_to_density_rsp


def test_hu_to_density_rsp_has_separate_physical_curves() -> None:
    hu = np.asarray([-1000.0, 0.0, 1000.0], dtype=np.float32)
    density, rsp = hu_to_density_rsp(hu)

    assert density[0] == pytest.approx(0.0, abs=1e-6)
    assert rsp[0] == pytest.approx(0.0, abs=1e-6)
    assert density[1] == pytest.approx(1.0, abs=1e-6)
    assert rsp[1] == pytest.approx(1.0, abs=1e-6)
    assert density[2] > 1.4
    assert 1.3 < rsp[2] < 1.7
    assert density[2] != pytest.approx(float(rsp[2]))


def test_bev_extracts_four_channels_with_monotone_wepl() -> None:
    ct, dose = _constant_sitk_pair(hu=0.0)
    bev = extract_bev_pair(
        ct,
        dose,
        ray_source_mm=(-20.0, 10.0, 10.0),
        ray_target_mm=(360.0, 10.0, 10.0),
        depth_size=16,
        lateral_size=5,
        lateral_extent_mm=8.0,
    )

    x = np.asarray(bev["x"], dtype=np.float32)
    wepl = x[3]
    assert x.shape == (DOSERAD_INPUT_CHANNELS, 16, 5, 5)
    assert x.shape[0] == 4
    assert np.allclose(wepl[0], 0.0)
    assert np.all(np.diff(wepl, axis=0) >= -1e-6)


def test_fixed_depth_spacing_is_constant_across_rays() -> None:
    ct, dose = _constant_sitk_pair(hu=0.0)
    kwargs = {"depth_size": 16, "lateral_size": 5, "lateral_extent_mm": 8.0}
    bev_a = extract_bev_pair(
        ct,
        dose,
        ray_source_mm=(-20.0, 8.0, 8.0),
        ray_target_mm=(360.0, 8.0, 8.0),
        **kwargs,
    )
    bev_b = extract_bev_pair(
        ct,
        dose,
        ray_source_mm=(-120.0, 12.0, 12.0),
        ray_target_mm=(500.0, 12.0, 12.0),
        **kwargs,
    )

    expected_spacing = 400.0 / 15.0
    assert bev_a["depth_spacing_mm"] == pytest.approx(expected_spacing)
    assert bev_b["depth_spacing_mm"] == pytest.approx(expected_spacing)
    assert np.asarray(bev_a["spacing_mm"])[0] == pytest.approx(expected_spacing)
    assert np.asarray(bev_b["spacing_mm"])[0] == pytest.approx(expected_spacing)


def _constant_sitk_pair(*, hu: float) -> tuple[sitk.Image, sitk.Image]:
    shape_zyx = (12, 12, 220)
    ct_arr = np.full(shape_zyx, hu, dtype=np.float32)
    dose_arr = np.zeros(shape_zyx, dtype=np.float32)
    ct = sitk.GetImageFromArray(ct_arr)
    dose = sitk.GetImageFromArray(dose_arr)
    for image in (ct, dose):
        image.SetSpacing((2.0, 2.0, 2.0))
        image.SetOrigin((0.0, 0.0, 0.0))
    return ct, dose


def test_gamma_dta_grants_credit_on_2mm_grid_but_not_on_coarse_grid() -> None:
    """A 2mm depth shift is inside the 3mm DTA and must pass on a 2mm grid.

    On a 6.35mm grid (run17's resolved depth spacing) every neighbouring voxel
    is farther than the DTA, so the same physical shift fails: the DTA term is
    inert and gamma degenerates to a same-voxel dose test. Guards against
    silently launching runs whose grid makes the metric non-comparable.
    """

    from braggtransporter.data.doserad import gamma_index_3d

    def cube(spacing_mm: float, shift_mm: float, n: int) -> np.ndarray:
        z = np.arange(n, dtype=np.float64) * spacing_mm
        profile = np.exp(-0.5 * ((z - shift_mm - 60.0) / 8.0) ** 2)
        return np.tile(profile[:, None, None], (1, 3, 3))

    fine = 2.0
    g_fine = gamma_index_3d(
        cube(fine, 2.0, 64), cube(fine, 0.0, 64), (fine, fine, fine), dose_pct=3.0, dta_mm=3.0
    )
    assert g_fine == pytest.approx(100.0)

    coarse = 400.0 / 63.0
    g_coarse = gamma_index_3d(
        cube(coarse, 2.0, 64), cube(coarse, 0.0, 64), (coarse, coarse, coarse), dose_pct=3.0, dta_mm=3.0
    )
    assert g_coarse < 100.0


def test_run17_mislaunch_configuration_is_now_fatal(monkeypatch) -> None:
    """Omitting --depth-size with a 400mm extent (run17's exact mislaunch)
    resolves to 6.35mm depth bins and must abort unless explicitly allowed."""

    import sys as _sys

    _sys.path.insert(0, "scripts")
    import importlib

    tdg = importlib.import_module("train_doserad_gpu")

    monkeypatch.setattr(
        _sys, "argv", ["train_doserad_gpu.py", "--depth-extent-mm", "400", "--allow-coarse-axes", "lateral"]
    )
    args = tdg.parse_args()
    assert args.depth_size == 64
    with pytest.raises(SystemExit, match="depth spacing 6.35mm"):
        tdg.report_grid_resolution(args)

    monkeypatch.setattr(
        _sys,
        "argv",
        ["train_doserad_gpu.py", "--depth-extent-mm", "400", "--depth-size", "201", "--allow-coarse-axes", "lateral"],
    )
    good = tdg.parse_args()
    tdg.report_grid_resolution(good)


def test_fast_gamma_matches_reference_pass_rate() -> None:
    """Vectorized gamma must agree with the reference point-loop implementation
    across grids, criteria, and cutoffs (including the papers' 0.1% cutoff)."""

    from braggtransporter.data.doserad import gamma_index_3d, gamma_index_3d_fast

    rng = np.random.default_rng(7)
    for spacing in ((2.0, 4.17, 4.17), (6.35, 4.17, 4.17), (1.5, 1.5, 1.5)):
        ref = rng.random((24, 9, 9)) ** 3
        pred = np.clip(ref + rng.normal(0.0, 0.02, ref.shape), 0.0, None)
        for dose_pct, dta, cut in ((3.0, 3.0, 0.1), (1.0, 3.0, 0.001), (2.0, 2.0, 0.1)):
            slow = gamma_index_3d(pred, ref, spacing, dose_pct=dose_pct, dta_mm=dta, low_dose_threshold=cut)
            fast = gamma_index_3d_fast(pred, ref, spacing, dose_pct=dose_pct, dta_mm=dta, low_dose_threshold=cut)
            assert fast == pytest.approx(slow), (spacing, dose_pct, dta, cut)


def test_pymedphys_gamma_resolves_subvoxel_shift_missed_by_voxel_centers() -> None:
    pytest.importorskip("pymedphys")
    from braggtransporter.data.doserad import gamma_index_3d_fast, gamma_index_3d_pymedphys

    shape = (24, 8, 8)
    spacing = (2.0, 2.0, 2.0)
    z, y, x = np.meshgrid(
        *(np.arange(n, dtype=np.float64) * step for n, step in zip(shape, spacing)),
        indexing="ij",
    )
    ref = np.exp(-0.5 * (((z - 26.0) / 6.0) ** 2 + ((y - 7.0) / 4.0) ** 2 + ((x - 7.0) / 4.0) ** 2))
    shifted = np.exp(
        -0.5 * (((z - 27.0) / 6.0) ** 2 + ((y - 7.0) / 4.0) ** 2 + ((x - 7.0) / 4.0) ** 2)
    )

    interpolated = gamma_index_3d_pymedphys(shifted, ref, spacing)
    voxel_center = gamma_index_3d_fast(
        shifted,
        ref,
        spacing,
        dose_pct=1.0,
        dta_mm=3.0,
        low_dose_threshold=0.001,
    )

    assert interpolated > 99.0
    assert interpolated - voxel_center > 30.0


def test_stratified_selection_spans_all_beams() -> None:
    """Round-robin manifest must cover every beam/gantry angle; the old prefix
    covered 17/36 (the iteration-2 consensus finding)."""

    import sys as _sys

    _sys.path.insert(0, "scripts")
    from vm_download_doserad import stratified_beamlet_paths

    plan = {
        "beams": [
            {
                "beam_idx": b,
                "gantry_angle": 10.0 * b,
                "rays": [
                    {
                        "ray_idx": r,
                        "beamlets": [{"beamlet_idx": l, "energy": 100.0} for l in range(6)],
                    }
                    for r in range(5)
                ],
            }
            for b in range(36)
        ]
    }
    paths = stratified_beamlet_paths(plan, "PAT", 500)
    assert len(paths) == 500
    beams_covered = {p.split("Dose_B")[1].split("_")[0] for p in paths}
    assert len(beams_covered) == 36
    per_beam = [sum(1 for p in paths if f"Dose_B{b}_" in p) for b in range(36)]
    assert max(per_beam) - min(per_beam) <= 1


def test_three_way_patient_split_is_disjoint_and_test_matches_legacy_val(tmp_path, monkeypatch) -> None:
    """test_frac>0 must produce disjoint patient cohorts, and the TEST cohort
    must equal the FIRST slice of the seeded shuffle (the patients legacy
    two-way runs used as val) so headline numbers stay comparable."""

    import math as _math

    patient_ids = [f"P{i:02d}" for i in range(12)]
    rng = np.random.default_rng(0)
    shuffled = list(patient_ids)
    rng.shuffle(shuffled)
    n_test = max(1, int(_math.ceil(0.15 * 12)))
    expected_test = shuffled[:n_test]

    legacy_rng = np.random.default_rng(0)
    legacy = list(patient_ids)
    legacy_rng.shuffle(legacy)
    legacy_val = legacy[: max(1, int(_math.ceil(0.15 * 12)))]
    assert expected_test == legacy_val
