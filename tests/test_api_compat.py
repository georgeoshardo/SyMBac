import importlib.util
import sys
import types

import pytest

if importlib.util.find_spec("napari") is None:
    napari_stub = types.ModuleType("napari")
    napari_stub.Viewer = object
    napari_stub.run = lambda: None
    sys.modules.setdefault("napari", napari_stub)

from SyMBac import cell_simulation
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


def _run_simulation_kwargs(tmp_path):
    return {
        "trench_length": 15,
        "trench_width": 1.5,
        "cell_max_length": 6.0,
        "cell_width": 1.0,
        "sim_length": 1,
        "pix_mic_conv": 0.065,
        "gravity": 0,
        "phys_iters": 1,
        "max_length_std": 0.2,
        "width_std": 0.1,
        "save_dir": str(tmp_path / "sim"),
        "resize_amount": 1,
        "lysis_p": 0.0,
        "show_window": False,
    }


def test_simulation_legacy_aliases_are_supported(tmp_path):
    kwargs = _simulation_kwargs(tmp_path)
    kwargs.pop("max_length_std")
    kwargs.pop("width_std")
    with pytest.warns(FutureWarning, match="2026-09-01"):
        simulation = Simulation(**kwargs, max_length_var=0.4, width_var=0.2)

    assert simulation.max_length_std == pytest.approx(0.4)
    assert simulation.width_std == pytest.approx(0.2)
    assert simulation.max_length_var == simulation.max_length_std
    assert simulation.width_var == simulation.width_std


def test_simulation_rejects_conflicting_alias_values(tmp_path):
    kwargs = _simulation_kwargs(tmp_path)
    with pytest.raises(ValueError, match="max_length_std"):
        Simulation(**kwargs, max_length_var=0.3)


def test_run_simulation_legacy_aliases_are_supported(tmp_path, monkeypatch):
    captured = {}

    def fake_run_simulation_impl(**kwargs):
        captured.update(kwargs)
        return ["cells"], "space", ["historic"]

    monkeypatch.setattr(cell_simulation, "_run_simulation_impl", fake_run_simulation_impl)

    kwargs = _run_simulation_kwargs(tmp_path)
    kwargs.pop("max_length_std")
    kwargs.pop("width_std")
    with pytest.warns(FutureWarning, match="2026-09-01"):
        result = cell_simulation.run_simulation(**kwargs, max_length_var=0.5, width_var=0.25)

    assert result == (["cells"], "space", ["historic"])
    assert captured["max_length_std"] == pytest.approx(0.5)
    assert captured["width_std"] == pytest.approx(0.25)
    assert captured["max_length_std_is_legacy"] is True
    assert captured["width_std_is_legacy"] is True


def test_run_simulation_rejects_conflicting_alias_values(tmp_path):
    kwargs = _run_simulation_kwargs(tmp_path)
    with pytest.raises(ValueError, match="width_std"):
        cell_simulation.run_simulation(**kwargs, width_var=0.3)


def test_legacy_impl_uses_scale_factor_for_std_conversion(tmp_path, monkeypatch):
    captured = {}

    class DummyCell:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_step_and_update(**kwargs):
        return kwargs["cells"]

    monkeypatch.setattr(cell_simulation, "Cell", DummyCell)
    monkeypatch.setattr(cell_simulation, "trench_creator", lambda *args, **kwargs: None)
    monkeypatch.setattr(cell_simulation, "step_and_update", fake_step_and_update)

    cell_simulation._run_simulation_impl(
        trench_length=15,
        trench_width=1.5,
        cell_max_length=6.0,
        cell_width=1.0,
        sim_length=0,
        pix_mic_conv=0.065,
        gravity=0,
        phys_iters=1,
        max_length_std=0.2,
        width_std=0.1,
        save_dir=str(tmp_path / "sim"),
        resize_amount=1,
        lysis_p=0.0,
        show_window=False,
    )

    scale_factor = (1 / 0.065) * 1
    assert captured["max_length_std"] == pytest.approx(0.2 * scale_factor)
    assert captured["width_std"] == pytest.approx(0.1 * scale_factor)


def test_legacy_impl_uses_sqrt_scale_factor_for_legacy_aliases(tmp_path, monkeypatch):
    captured = {}

    class DummyCell:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_step_and_update(**kwargs):
        return kwargs["cells"]

    monkeypatch.setattr(cell_simulation, "Cell", DummyCell)
    monkeypatch.setattr(cell_simulation, "trench_creator", lambda *args, **kwargs: None)
    monkeypatch.setattr(cell_simulation, "step_and_update", fake_step_and_update)

    cell_simulation._run_simulation_impl(
        trench_length=15,
        trench_width=1.5,
        cell_max_length=6.0,
        cell_width=1.0,
        sim_length=0,
        pix_mic_conv=0.065,
        gravity=0,
        phys_iters=1,
        max_length_std=0.2,
        width_std=0.1,
        save_dir=str(tmp_path / "sim"),
        resize_amount=1,
        lysis_p=0.0,
        show_window=False,
        max_length_std_is_legacy=True,
        width_std_is_legacy=True,
    )

    scale_factor = (1 / 0.065) * 1
    assert captured["max_length_std"] == pytest.approx(0.2 * (scale_factor**0.5))
    assert captured["width_std"] == pytest.approx(0.1 * (scale_factor**0.5))


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
