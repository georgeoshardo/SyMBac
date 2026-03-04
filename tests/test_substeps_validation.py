import importlib.util
import sys
import types

import pytest

if importlib.util.find_spec("napari") is None:
    napari_stub = types.ModuleType("napari")
    napari_stub.Viewer = object
    napari_stub.run = lambda: None
    sys.modules.setdefault("napari", napari_stub)

from SyMBac.simulation import Simulation


def _simulation_kwargs(tmp_path):
    return {
        "trench_length": 15,
        "trench_width": 1.5,
        "cell_max_length": 6.0,
        "max_length_std": 0.2,
        "cell_width": 1.0,
        "width_std": 0.1,
        "lysis_p": 0.0,
        "sim_length": 1,
        "pix_mic_conv": 0.065,
        "gravity": 0,
        "phys_iters": 1,
        "resize_amount": 1,
        "save_dir": str(tmp_path / "sim"),
    }


@pytest.mark.parametrize("bad_substeps", [True, False, 0, -1, 1.5, 2.0])
def test_substeps_validation_rejects_invalid_values(tmp_path, bad_substeps):
    kwargs = _simulation_kwargs(tmp_path)
    with pytest.raises(ValueError, match="substeps"):
        Simulation(**kwargs, substeps=bad_substeps)


def test_substeps_validation_accepts_positive_int(tmp_path):
    kwargs = _simulation_kwargs(tmp_path)
    simulation = Simulation(**kwargs, substeps=3)
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
    kwargs = _simulation_kwargs(tmp_path)
    kwargs[field_name] = bad_value
    with pytest.raises(TypeError, match=f"{field_name}"):
        Simulation(**kwargs)


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("brownian_longitudinal_std", -0.1),
        ("brownian_transverse_std", -0.1),
        ("brownian_rotation_std", -0.01),
        ("brownian_persistence", -0.1),
        ("brownian_persistence", 1.0),
        ("brownian_max_dx_fraction_of_trench_width", 0.0),
        ("brownian_max_dx_fraction_of_trench_width", -0.1),
        ("brownian_max_dy_fraction_of_segment_radius", 0.0),
        ("brownian_max_dy_px_floor", 0.0),
        ("brownian_max_dtheta", 0.0),
        ("brownian_backoff_attempts", 0),
        ("brownian_backoff_attempts", 1.5),
    ],
)
def test_brownian_validation_rejects_invalid_ranges(tmp_path, field_name, bad_value):
    kwargs = _simulation_kwargs(tmp_path)
    kwargs[field_name] = bad_value
    with pytest.raises(ValueError):
        Simulation(**kwargs)
