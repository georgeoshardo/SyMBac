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
from pymunk.vec2d import Vec2d


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


class _DummyBody:
    def __init__(self, x=0.0, y=0.0, angle=0.0):
        self.position = Vec2d(x, y)
        self.angle = angle


class _DummySegment:
    def __init__(self, x=0.0, y=0.0, angle=0.0):
        self.body = _DummyBody(x=x, y=y, angle=angle)
        self.radius = 0.5


class _DummyPhysicsRepresentation:
    def __init__(self, x=35.0, y=10.0):
        self.segments = [_DummySegment(x=x, y=y, angle=0.0)]


class _DummyCellWithPhysics(_DummyCell):
    def __init__(self, group_id, x=35.0, y=10.0):
        super().__init__(group_id)
        self.physics_representation = _DummyPhysicsRepresentation(x=x, y=y)


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


def test_run_simulation_applies_low_level_config_overrides(tmp_path, monkeypatch):
    captured = {}

    class _CapturingSimulator:
        def __init__(self, **kwargs):
            captured["physics_config"] = kwargs["physics_config"]
            captured["cell_config"] = kwargs["initial_cell_config"]
            self.space = {"dummy_space": True}
            self.colony = _DummyColony()

        def step(self):
            return None

    monkeypatch.setattr(physics_simulator_module, "Simulator", _CapturingSimulator)
    monkeypatch.setattr(cell_snapshot_module, "CellSnapshot", _DummySnapshot)

    kwargs = _simulation_kwargs(tmp_path)
    simulation = Simulation(
        **kwargs,
        cell_config_overrides={
            "MAX_BEND_ANGLE": 0.02,
            "STIFFNESS": 123456.0,
            "PIVOT_JOINT_STIFFNESS": 4321.0,
            "NOISE_STRENGTH": 0.09,
        },
        physics_config_overrides={
            "ITERATIONS": 77,
            "DAMPING": 0.35,
        },
    )
    simulation.run_simulation(show_window=False)

    cell_cfg = captured["cell_config"]
    physics_cfg = captured["physics_config"]
    assert cell_cfg.MAX_BEND_ANGLE == 0.02
    assert cell_cfg.STIFFNESS == 123456.0
    assert cell_cfg.PIVOT_JOINT_STIFFNESS == 4321.0
    assert cell_cfg.NOISE_STRENGTH == 0.09
    assert physics_cfg.ITERATIONS == 77
    assert physics_cfg.DAMPING == 0.35


def test_run_simulation_brownian_jitter_moves_cells(tmp_path, monkeypatch):
    captured = {}

    class _JitterColony:
        def __init__(self):
            self.cells = [_DummyCellWithPhysics(1)]

        def delete_cell(self, cell):
            self.cells.remove(cell)

    class _JitterSimulator:
        def __init__(self, **_kwargs):
            self.space = {"dummy_space": True}
            self.colony = _JitterColony()
            captured["cell"] = self.colony.cells[0]

        def step(self):
            return None

    monkeypatch.setattr(physics_simulator_module, "Simulator", _JitterSimulator)
    monkeypatch.setattr(cell_snapshot_module, "CellSnapshot", _DummySnapshot)
    monkeypatch.setattr(
        simulation_module.np.random,
        "normal",
        lambda loc, scale: scale if scale > 0 else 0.0,
    )

    kwargs = _simulation_kwargs(tmp_path)
    simulation = Simulation(
        **kwargs,
        brownian_longitudinal_std=0.05,
        brownian_transverse_std=0.02,
        brownian_rotation_std=0.01,
        brownian_persistence=0.0,
    )
    simulation.run_simulation(show_window=False)

    body = captured["cell"].physics_representation.segments[0].body
    assert float(body.position[0]) != 35.0 or float(body.position[1]) != 10.0
    assert float(body.angle) != 0.0


def test_run_simulation_brownian_jitter_rolls_back_if_out_of_bounds(tmp_path, monkeypatch):
    captured = {}

    class _BoundaryJitterColony:
        def __init__(self):
            # Near right wall so positive x-jitter should violate bounds.
            self.cells = [_DummyCellWithPhysics(1, x=46.0, y=10.0)]

        def delete_cell(self, cell):
            self.cells.remove(cell)

    class _BoundaryJitterSimulator:
        def __init__(self, **_kwargs):
            self.space = {"dummy_space": True}
            self.colony = _BoundaryJitterColony()
            captured["cell"] = self.colony.cells[0]

        def step(self):
            return None

    monkeypatch.setattr(physics_simulator_module, "Simulator", _BoundaryJitterSimulator)
    monkeypatch.setattr(cell_snapshot_module, "CellSnapshot", _DummySnapshot)
    monkeypatch.setattr(
        simulation_module.np.random,
        "normal",
        lambda loc, scale: scale if scale > 0 else 0.0,
    )

    kwargs = _simulation_kwargs(tmp_path)
    simulation = Simulation(
        **kwargs,
        brownian_longitudinal_std=0.0,
        brownian_transverse_std=5.0,
        brownian_rotation_std=0.02,
        brownian_persistence=0.0,
    )
    simulation.run_simulation(show_window=False)

    body = captured["cell"].physics_representation.segments[0].body
    assert float(body.position[0]) == pytest.approx(46.0)
    assert float(body.position[1]) == pytest.approx(10.0)
    assert float(body.angle) == pytest.approx(0.0)
