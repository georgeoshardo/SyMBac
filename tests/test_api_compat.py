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
        "substeps": 1,
    }


def test_simulation_constructor_requires_keyword_arguments_after_cell_max_length(tmp_path):
    kwargs = _simulation_kwargs(tmp_path)
    with pytest.raises(TypeError):
        Simulation(
            kwargs["trench_length"],
            kwargs["trench_width"],
            kwargs["cell_max_length"],
            kwargs["max_length_std"],
            kwargs["cell_width"],
            kwargs["width_std"],
            kwargs["lysis_p"],
            kwargs["sim_length"],
            kwargs["pix_mic_conv"],
            kwargs["gravity"],
            kwargs["phys_iters"],
            kwargs["resize_amount"],
            kwargs["save_dir"],
            kwargs["substeps"],
        )


def test_simulation_rejects_removed_legacy_aliases(tmp_path):
    kwargs = _simulation_kwargs(tmp_path)
    with pytest.raises(TypeError, match="max_length_var"):
        Simulation(**kwargs, max_length_var=0.3)

    with pytest.raises(TypeError, match="width_var"):
        Simulation(**kwargs, width_var=0.2)


def test_draw_simulation_opl_rejects_removed_do_transformation_argument(tmp_path):
    kwargs = _simulation_kwargs(tmp_path)
    simulation = Simulation(**kwargs)

    with pytest.raises(TypeError, match="do_transformation"):
        simulation.draw_simulation_OPL(do_transformation=False)
