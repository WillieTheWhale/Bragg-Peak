from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from braggtransporter.models.dota3d import DoTA3D


def test_run20_config_is_isotropic_disjoint_and_matches_model() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "doserad_run20.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    coords = config["coordinates"]
    for axis in ("depth", "lateral_y", "lateral_x"):
        resolved = float(coords[axis]["center_extent_mm"]) / (int(coords[axis]["bins"]) - 1)
        assert resolved == pytest.approx(2.0)
        assert float(coords[axis]["spacing_mm"]) == pytest.approx(resolved)

    split = config["dataset"]["split"]
    train = set(split["train_patients"])
    val = set(split["validation_patients"])
    test = set(split["untouched_test_patients"])
    assert not train & val
    assert not train & test
    assert not val & test
    assert train | val | test == set(config["dataset"]["patients"])

    model_config = config["model"]
    model = DoTA3D(
        d_model=int(model_config["token_dimension"]),
        n_layers=int(model_config["transformer_layers"]),
        n_heads=int(model_config["attention_heads"]),
        d_ff=int(model_config["feedforward_dimension"]),
        dropout=float(model_config["dropout"]),
        max_depth=int(coords["depth"]["bins"]),
        lateral_size=int(coords["lateral_x"]["bins"]),
        encoder_channels=int(model_config["encoder_channels"]),
    )
    assert model.param_count() == int(model_config["parameter_count"])

    paper = config["evaluation"]["final"]["criteria"][0]
    assert paper["engine"] == "PyMedPhys_global_gamma"
    assert paper["dose_difference_percent"] == 1.0
    assert paper["distance_to_agreement_mm"] == 3.0
    assert paper["cutoff_fraction"] == 0.001
