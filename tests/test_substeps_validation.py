import importlib.util
import sys
import types

import pytest
from pydantic import ValidationError

if importlib.util.find_spec("napari") is None:
    napari_stub = types.ModuleType("napari")
    napari_stub.Viewer = object
    napari_stub.run = lambda: None
    sys.modules.setdefault("napari", napari_stub)

from SyMBac.config_models import (
    BrownianJitterSpec,
    SimulationCellSpec,
    SimulationGeometrySpec,
    SimulationLowLevelSpec,
    SimulationPhysicsSpec,
    SimulationRuntimeSpec,
    SimulationSpec,
)
from SyMBac.simulation import Simulation


def _build_spec(
    tmp_path,
    *,
    substeps=1,
    cell_config_overrides=None,
    physics_config_overrides=None,
    brownian_overrides=None,
):
    return SimulationSpec(
        geometry=SimulationGeometrySpec(
            trench_length=15.0,
            trench_width=1.5,
            pix_mic_conv=0.065,
            resize_amount=1,
        ),
        cell=SimulationCellSpec(
            cell_max_length=6.0,
            max_length_std=0.2,
            cell_width=1.0,
            width_std=0.1,
            lysis_p=0.0,
        ),
        physics=SimulationPhysicsSpec(gravity=0.0, phys_iters=1),
        runtime=SimulationRuntimeSpec(
            sim_length=1,
            substeps=substeps,
            save_dir=str(tmp_path / "sim"),
        ),
        low_level=SimulationLowLevelSpec(
            cell_config_overrides=cell_config_overrides or {},
            physics_config_overrides=physics_config_overrides or {},
        ),
        brownian=BrownianJitterSpec(**(brownian_overrides or {})),
    )


@pytest.mark.parametrize("bad_substeps", [True, False, 0, -1, 1.5, 2.0, "3"])
def test_substeps_validation_rejects_invalid_values(tmp_path, bad_substeps):
    with pytest.raises(ValidationError, match="substeps"):
        _build_spec(tmp_path, substeps=bad_substeps)


def test_substeps_validation_accepts_positive_int(tmp_path):
    simulation = Simulation(_build_spec(tmp_path, substeps=3))
    assert simulation.substeps == 3


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("cell_config_overrides", 1),
        ("cell_config_overrides", []),
        ("physics_config_overrides", 1.5),
        ("physics_config_overrides", ()),
    ],
)
def test_config_overrides_validation_rejects_non_dict(tmp_path, field_name, bad_value):
    kwargs = {field_name: bad_value}
    with pytest.raises(ValidationError, match=field_name):
        SimulationLowLevelSpec(**kwargs)


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("longitudinal_std", -0.1),
        ("transverse_std", -0.1),
        ("rotation_std", -0.01),
        ("persistence", -0.1),
        ("persistence", 1.0),
        ("max_dx_fraction_of_trench_width", 0.0),
        ("max_dx_fraction_of_trench_width", -0.1),
        ("max_dy_fraction_of_segment_radius", 0.0),
        ("max_dy_px_floor", 0.0),
        ("max_dtheta", 0.0),
        ("backoff_attempts", 0),
        ("backoff_attempts", 1.5),
        ("application_mode", "not-a-mode"),
        ("projection_angular_damping", -0.1),
        ("projection_angular_damping", 1.1),
    ],
)
def test_brownian_validation_rejects_invalid_ranges(tmp_path, field_name, bad_value):
    with pytest.raises(ValidationError):
        _build_spec(tmp_path, brownian_overrides={field_name: bad_value})


@pytest.mark.parametrize("mode", ["teleport", "velocity", "impulse"])
def test_brownian_application_mode_accepts_valid_values(tmp_path, mode):
    simulation = Simulation(
        _build_spec(
            tmp_path,
            brownian_overrides={
                "application_mode": mode,
                "projection_angular_damping": 0.5,
            },
        )
    )
    assert simulation.brownian_application_mode == mode
    assert simulation.brownian_projection_angular_damping == pytest.approx(0.5)
