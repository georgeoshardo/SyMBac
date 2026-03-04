import importlib.util
import os
import pickle
import sys
import types

import pytest

if importlib.util.find_spec("napari") is None:
    napari_stub = types.ModuleType("napari")
    napari_stub.Viewer = object
    napari_stub.run = lambda: None
    sys.modules.setdefault("napari", napari_stub)

from SyMBac.simulation import Simulation
import SyMBac.simulation as simulation_module
import SyMBac.cell_snapshot as cell_snapshot_module
import SyMBac.cell_simulation as cell_simulation_module
import SyMBac.physics.simulator as physics_simulator_module


class _DummyCell:
    def __init__(self, group_id):
        self.group_id = group_id


class _DummyColony:
    def __init__(self):
        self.cells = [_DummyCell(1)]

    def delete_cell(self, cell):
        self.cells.remove(cell)


class _DummySimulator:
    def __init__(self, **kwargs):
        self.space = {"dummy_space": True}
        self.colony = _DummyColony()

    def step(self):
        return None


class _DummySnapshot:
    def __init__(
        self,
        simcell,
        t=0,
        mother_mask_label=None,
        generation=0,
        just_divided=False,
        lysis_p=0.0,
    ):
        self.mask_label = simcell.group_id
        self.segment_positions = []
        self.t = t
        self.generation = generation
        self.mother_mask_label = mother_mask_label
        self.just_divided = just_divided
        self.lysis_p = lysis_p


def _simulation_kwargs(tmp_path):
    return {
        "trench_length": 15,
        "trench_width": 1.5,
        "cell_max_length": 6.0,
        "max_length_std": 0.2,
        "cell_width": 1.0,
        "width_std": 0.1,
        "lysis_p": 0.0,
        "sim_length": 2,
        "pix_mic_conv": 0.065,
        "gravity": 0,
        "phys_iters": 1,
        "resize_amount": 1,
        "save_dir": str(tmp_path / "sim"),
        "substeps": 1,
    }


def test_run_simulation_persists_required_pickles(tmp_path, monkeypatch):
    monkeypatch.setattr(physics_simulator_module, "Simulator", _DummySimulator)
    monkeypatch.setattr(cell_snapshot_module, "CellSnapshot", _DummySnapshot)

    replace_calls = []
    real_replace = os.replace

    def tracking_replace(src, dst):
        replace_calls.append((src, dst))
        return real_replace(src, dst)

    monkeypatch.setattr(simulation_module.os, "replace", tracking_replace)

    simulation = Simulation(**_simulation_kwargs(tmp_path))
    simulation.run_simulation(show_window=False)

    save_dir = tmp_path / "sim"
    cell_path = save_dir / "cell_timeseries.p"
    space_path = save_dir / "space_timeseries.p"

    assert cell_path.exists()
    assert space_path.exists()

    with open(cell_path, "rb") as handle:
        stored_cell_timeseries = pickle.load(handle)
    with open(space_path, "rb") as handle:
        stored_space = pickle.load(handle)

    assert len(stored_cell_timeseries) == simulation.sim_length
    assert stored_space == simulation.space

    replace_targets = {dst for _, dst in replace_calls}
    assert str(cell_path) in replace_targets
    assert str(space_path) in replace_targets
    assert list(save_dir.glob(".tmp_*.p")) == []


def test_load_sim_dir_missing_artifacts_error_names_expected_files(tmp_path):
    load_dir = tmp_path / "load_artifacts"
    load_dir.mkdir()
    with open(load_dir / "cell_timeseries.p", "wb") as handle:
        pickle.dump(["cells"], handle)

    kwargs = _simulation_kwargs(tmp_path)
    kwargs["load_sim_dir"] = str(load_dir)
    kwargs["save_dir"] = str(tmp_path / "save_target")

    with pytest.raises(FileNotFoundError) as exc:
        Simulation(**kwargs)

    message = str(exc.value)
    assert "cell_timeseries.p" in message
    assert "space_timeseries.p" in message


def test_load_sim_dir_reads_existing_artifacts(tmp_path):
    load_dir = tmp_path / "load_complete"
    load_dir.mkdir()
    expected_cells = [["frame"]]
    expected_space = {"space": 1}
    with open(load_dir / "cell_timeseries.p", "wb") as handle:
        pickle.dump(expected_cells, handle)
    with open(load_dir / "space_timeseries.p", "wb") as handle:
        pickle.dump(expected_space, handle)

    kwargs = _simulation_kwargs(tmp_path)
    kwargs["load_sim_dir"] = str(load_dir)
    kwargs["save_dir"] = str(tmp_path / "save_target")

    simulation = Simulation(**kwargs)
    assert simulation.cell_timeseries == expected_cells
    assert simulation.space == expected_space


def test_cell_simulation_run_simulation_persists_required_pickles(tmp_path):
    save_dir = tmp_path / "legacy_sim"
    save_dir.mkdir()
    result = cell_simulation_module.run_simulation(
        trench_length=15,
        trench_width=1.5,
        cell_max_length=6.0,
        cell_width=1.0,
        sim_length=2,
        pix_mic_conv=0.065,
        gravity=0,
        phys_iters=1,
        max_length_std=0.2,
        width_std=0.1,
        save_dir=str(save_dir),
        resize_amount=1,
        lysis_p=0.0,
        show_window=False,
    )

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert (save_dir / "cell_timeseries.p").exists()
    assert (save_dir / "space_timeseries.p").exists()
