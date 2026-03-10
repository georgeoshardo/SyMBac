import importlib.util
import os
import pickle
import sys
import types

import numpy as np
import pytest

if importlib.util.find_spec("napari") is None:
    napari_stub = types.ModuleType("napari")
    napari_stub.Viewer = object
    napari_stub.run = lambda: None
    sys.modules.setdefault("napari", napari_stub)

from SyMBac.simulation import Simulation
import SyMBac.simulation as simulation_module
import SyMBac.cell_snapshot as cell_snapshot_module
import SyMBac.physics.simulator as physics_simulator_module
from SyMBac.physics.microfluidic_geometry import (
    Bounds2D,
    GeometryLayout,
    GeometrySpec,
    SegmentPrimitive,
    TrenchGeometrySpec,
)
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
        self.ID = simcell.group_id
        self.segment_positions = np.array([[0.0, 0.0]], dtype=np.float64)
        self.segment_radii = np.array([0.5], dtype=np.float64)
        self.t = t
        self.generation = generation
        self.mother_mask_label = mother_mask_label
        self.just_divided = just_divided
        self.lysis_p = lysis_p

    def to_segment_dict(self):
        return {
            "positions": self.segment_positions,
            "radii": self.segment_radii,
            "mask_label": self.mask_label,
            "cell_id": self.ID,
        }


class _StoredSnapshot:
    def __init__(self, mask_label=1, positions=((5.0, 7.0),), radii=(1.0,), t=0, mother_mask_label=None):
        self.mask_label = mask_label
        self.ID = mask_label
        self.segment_positions = np.array(positions, dtype=np.float64)
        self.segment_radii = np.array(radii, dtype=np.float64)
        self.t = t
        self.mother_mask_label = mother_mask_label

    def to_segment_dict(self):
        return {
            "positions": self.segment_positions,
            "radii": self.segment_radii,
            "mask_label": self.mask_label,
            "cell_id": self.ID,
        }


class _LegacyStoredCell:
    def __init__(self, mask_label=1, t=0):
        self.mask_label = mask_label
        self.t = t


class _DummyBody:
    def __init__(self, x=0.0, y=0.0, angle=0.0):
        self.position = Vec2d(x, y)
        self.angle = angle
        self.velocity = Vec2d(0.0, 0.0)
        self.angular_velocity = 0.0
        self.mass = 1.0
        self.moment = 1.0

    def apply_impulse_at_local_point(self, impulse):
        impulse_vec = Vec2d(float(impulse[0]), float(impulse[1]))
        self.velocity = self.velocity + (impulse_vec / self.mass)


class _DummySegment:
    def __init__(self, x=0.0, y=0.0, angle=0.0):
        self.body = _DummyBody(x=x, y=y, angle=angle)
        self.radius = 0.5


class _DummyPhysicsRepresentation:
    def __init__(self, x=0.0, y=0.0):
        self.segments = [_DummySegment(x=x, y=y, angle=0.0)]


class _DummyCellWithPhysics(_DummyCell):
    def __init__(self, group_id, x=0.0, y=0.0):
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


def _scale_factor(kwargs):
    return (1 / kwargs["pix_mic_conv"]) * kwargs["resize_amount"]


def _default_geometry_layout(kwargs):
    scale_factor = _scale_factor(kwargs)
    geometry = TrenchGeometrySpec(
        width=kwargs["trench_width"] * scale_factor,
        trench_length=kwargs["trench_length"] * scale_factor,
    )
    return geometry, GeometryLayout(geometry)


def _default_world_point(kwargs, local_x=0.0, local_y=10.0):
    _geometry, layout = _default_geometry_layout(kwargs)
    return layout.to_world_point((local_x, local_y))


class _CustomGeometrySpec(GeometrySpec):
    def __init__(self):
        self._local_bounds = Bounds2D(min_x=-5.0, min_y=-2.0, max_x=7.0, max_y=14.0)

    @property
    def local_bounds(self):
        return self._local_bounds

    @property
    def default_padding_x(self):
        return 12.0

    @property
    def default_padding_y(self):
        return 18.0

    @property
    def characteristic_width(self):
        return 12.0

    def build(self, space, layout):
        return None

    def preview_primitives(self, layout):
        p1 = layout.to_world_point((-5.0, 0.0))
        p2 = layout.to_world_point((7.0, 0.0))
        return [SegmentPrimitive(p1=p1, p2=p2, thickness=2.0)]

    def seed_cell_local_position(self, segment_radius):
        return (1.5, 6.0 + 0.5 * float(segment_radius))

    def positions_within_bounds(self, positions, radii, layout, *, enforce_open_end_cap):
        local_positions = layout.to_local_points(positions)
        for position, radius in zip(local_positions, radii):
            if float(position[0]) < (-5.0 + float(radius)):
                return False
            if float(position[0]) > (7.0 - float(radius)):
                return False
            if float(position[1]) < (0.0 + float(radius)):
                return False
            if enforce_open_end_cap and float(position[1]) > (14.0 - float(radius)):
                return False
        return True

    def cell_out_of_bounds(self, positions, radii, layout):
        return not self.positions_within_bounds(
            positions,
            radii,
            layout,
            enforce_open_end_cap=False,
        )

    def project_body_inside_bounds(self, body, radius, layout):
        local_position = layout.to_local_point((float(body.position[0]), float(body.position[1])))
        clamped_x = min(max(local_position[0], -5.0 + float(radius)), 7.0 - float(radius))
        clamped_y = min(max(local_position[1], 0.0 + float(radius)), 14.0 - float(radius))
        projected = (clamped_x != local_position[0]) or (clamped_y != local_position[1])
        if projected:
            body.position = Vec2d(*layout.to_world_point((clamped_x, clamped_y)))
        return projected, (clamped_x != local_position[0]), (clamped_y != local_position[1])


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
    expected_cells = [[_StoredSnapshot(mask_label=3, t=0)]]
    expected_space = {"space": 1}
    with open(load_dir / "cell_timeseries.p", "wb") as handle:
        pickle.dump(expected_cells, handle)
    with open(load_dir / "space_timeseries.p", "wb") as handle:
        pickle.dump(expected_space, handle)

    kwargs = _simulation_kwargs(tmp_path)
    kwargs["load_sim_dir"] = str(load_dir)
    kwargs["save_dir"] = str(tmp_path / "save_target")

    simulation = Simulation(**kwargs)
    loaded_snapshot = simulation.cell_timeseries[0][0]
    assert loaded_snapshot.mask_label == expected_cells[0][0].mask_label
    assert np.array_equal(loaded_snapshot.segment_positions, expected_cells[0][0].segment_positions)
    assert np.array_equal(loaded_snapshot.segment_radii, expected_cells[0][0].segment_radii)
    assert simulation.space == expected_space


def test_load_sim_dir_rejects_legacy_artifacts(tmp_path):
    load_dir = tmp_path / "load_legacy"
    load_dir.mkdir()
    with open(load_dir / "cell_timeseries.p", "wb") as handle:
        pickle.dump([[_LegacyStoredCell(mask_label=2, t=0)]], handle)
    with open(load_dir / "space_timeseries.p", "wb") as handle:
        pickle.dump({"space": 1}, handle)

    kwargs = _simulation_kwargs(tmp_path)
    kwargs["load_sim_dir"] = str(load_dir)
    kwargs["save_dir"] = str(tmp_path / "save_target")

    with pytest.raises(ValueError, match="Legacy simulation artifacts are no longer supported"):
        Simulation(**kwargs)


def test_draw_simulation_opl_renders_segment_snapshots_from_loaded_artifacts(tmp_path, monkeypatch):
    load_dir = tmp_path / "load_segments"
    load_dir.mkdir()
    expected_cells = [[_StoredSnapshot(mask_label=5, positions=((4.0, 6.0),), radii=(1.25,), t=0)]]
    with open(load_dir / "cell_timeseries.p", "wb") as handle:
        pickle.dump(expected_cells, handle)
    with open(load_dir / "space_timeseries.p", "wb") as handle:
        pickle.dump({"space": 1}, handle)

    monkeypatch.setattr(simulation_module, "get_trench_segments", lambda _space: "main_segments")

    kwargs = _simulation_kwargs(tmp_path)
    kwargs["load_sim_dir"] = str(load_dir)
    kwargs["save_dir"] = str(tmp_path / "save_target")

    simulation = Simulation(**kwargs)
    scenes, masks = simulation.draw_simulation_OPL(label_masks=True, return_output=True)

    assert simulation.main_segments == "main_segments"
    assert len(simulation.cell_timeseries_segments) == 1
    assert "cell_timeseries_properties" not in simulation.__dict__
    assert len(scenes) == 1
    assert len(masks) == 1
    assert scenes[0].shape == masks[0].shape


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


def test_run_simulation_default_geometry_uses_layout_derived_seed_position(tmp_path, monkeypatch):
    captured = {}

    class _CapturingSimulator:
        def __init__(self, **kwargs):
            captured["cell_config"] = kwargs["initial_cell_config"]
            self.space = {"dummy_space": True}
            self.colony = _DummyColony()

        def step(self):
            return None

    monkeypatch.setattr(physics_simulator_module, "Simulator", _CapturingSimulator)
    monkeypatch.setattr(cell_snapshot_module, "CellSnapshot", _DummySnapshot)

    kwargs = _simulation_kwargs(tmp_path)
    simulation = Simulation(**kwargs)
    simulation.run_simulation(show_window=False)

    expected_start = simulation._geometry_layout.to_world_point(
        simulation._geometry_spec.seed_cell_local_position(
            segment_radius=captured["cell_config"].SEGMENT_RADIUS
        )
    )
    assert captured["cell_config"].START_POS == pytest.approx(expected_start)


def test_run_simulation_accepts_custom_geometry_without_manual_world_coordinates(tmp_path, monkeypatch):
    captured = {}

    class _CapturingSimulator:
        def __init__(self, **kwargs):
            captured["cell_config"] = kwargs["initial_cell_config"]
            self.space = {"dummy_space": True}
            self.colony = _DummyColony()

        def step(self):
            return None

    monkeypatch.setattr(physics_simulator_module, "Simulator", _CapturingSimulator)
    monkeypatch.setattr(cell_snapshot_module, "CellSnapshot", _DummySnapshot)

    kwargs = _simulation_kwargs(tmp_path)
    custom_geometry = _CustomGeometrySpec()
    simulation = Simulation(**kwargs, geometry=custom_geometry)
    simulation.run_simulation(show_window=False)

    expected_layout = GeometryLayout(custom_geometry)
    expected_start = expected_layout.to_world_point(
        custom_geometry.seed_cell_local_position(
            segment_radius=captured["cell_config"].SEGMENT_RADIUS
        )
    )
    assert simulation._geometry_spec is custom_geometry
    assert simulation._geometry_layout.world_bounds.min_x == pytest.approx(expected_layout.padding_x)
    assert simulation._geometry_layout.world_bounds.min_y == pytest.approx(expected_layout.padding_y)
    assert captured["cell_config"].START_POS == pytest.approx(expected_start)


def test_run_simulation_brownian_jitter_moves_cells(tmp_path, monkeypatch):
    captured = {}
    kwargs = _simulation_kwargs(tmp_path)
    start_x, start_y = _default_world_point(kwargs)

    class _JitterColony:
        def __init__(self):
            self.cells = [_DummyCellWithPhysics(1, x=start_x, y=start_y)]

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

    simulation = Simulation(
        **kwargs,
        brownian_longitudinal_std=0.05,
        brownian_transverse_std=0.02,
        brownian_rotation_std=0.01,
        brownian_persistence=0.0,
    )
    simulation.run_simulation(show_window=False)

    body = captured["cell"].physics_representation.segments[0].body
    assert float(body.position[0]) != pytest.approx(start_x) or float(body.position[1]) != pytest.approx(start_y)
    assert float(body.angle) != 0.0


def test_run_simulation_brownian_jitter_rolls_back_if_out_of_bounds(tmp_path, monkeypatch):
    captured = {}
    kwargs = _simulation_kwargs(tmp_path)
    geometry, layout = _default_geometry_layout(kwargs)
    start_x, start_y = layout.to_world_point((geometry.inner_half_width - 0.5, 10.0))

    class _BoundaryJitterColony:
        def __init__(self):
            # Near right wall so positive x-jitter should violate bounds.
            self.cells = [_DummyCellWithPhysics(1, x=start_x, y=start_y)]

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

    simulation = Simulation(
        **kwargs,
        brownian_longitudinal_std=0.0,
        brownian_transverse_std=5.0,
        brownian_rotation_std=0.02,
        brownian_persistence=0.0,
    )
    simulation.run_simulation(show_window=False)

    body = captured["cell"].physics_representation.segments[0].body
    assert float(body.position[0]) == pytest.approx(start_x)
    assert float(body.position[1]) == pytest.approx(start_y)
    assert float(body.angle) == pytest.approx(0.0)


@pytest.mark.parametrize("brownian_mode", ["velocity", "impulse"])
def test_run_simulation_brownian_dynamic_mode_moves_cells(tmp_path, monkeypatch, brownian_mode):
    captured = {}
    kwargs = _simulation_kwargs(tmp_path)
    start_x, start_y = _default_world_point(kwargs)

    class _VelocityJitterColony:
        def __init__(self):
            self.cells = [_DummyCellWithPhysics(1, x=start_x, y=start_y)]

        def delete_cell(self, cell):
            self.cells.remove(cell)

    class _VelocityJitterSimulator:
        def __init__(self, **kwargs):
            self.space = {"dummy_space": True}
            self.colony = _VelocityJitterColony()
            self._dt = float(kwargs["physics_config"].DT)
            captured["cell"] = self.colony.cells[0]

        def step(self):
            for cell in self.colony.cells:
                for segment in cell.physics_representation.segments:
                    body = segment.body
                    body.position = body.position + body.velocity * self._dt
                    body.angle = float(body.angle) + float(body.angular_velocity) * self._dt

    monkeypatch.setattr(physics_simulator_module, "Simulator", _VelocityJitterSimulator)
    monkeypatch.setattr(cell_snapshot_module, "CellSnapshot", _DummySnapshot)
    monkeypatch.setattr(
        simulation_module.np.random,
        "normal",
        lambda loc, scale: scale if scale > 0 else 0.0,
    )

    simulation = Simulation(
        **kwargs,
        brownian_longitudinal_std=0.03,
        brownian_transverse_std=0.01,
        brownian_rotation_std=0.005,
        brownian_persistence=0.0,
        brownian_application_mode=brownian_mode,
    )
    simulation.run_simulation(show_window=False)

    body = captured["cell"].physics_representation.segments[0].body
    assert float(body.position[0]) != pytest.approx(start_x) or float(body.position[1]) != pytest.approx(start_y)
    assert float(body.angle) != pytest.approx(0.0)


def test_run_simulation_brownian_projection_clamps_out_of_bounds_velocity_mode(tmp_path, monkeypatch):
    captured = {}
    kwargs = _simulation_kwargs(tmp_path)
    geometry, layout = _default_geometry_layout(kwargs)
    start_x, start_y = layout.to_world_point((geometry.inner_half_width - 0.5, 10.0))

    class _BoundaryProjectionColony:
        def __init__(self):
            self.cells = [_DummyCellWithPhysics(1, x=start_x, y=start_y)]
            self.cells[0].physics_representation.segments[0].body.velocity = Vec2d(20.0, 0.0)
            self.cells[0].physics_representation.segments[0].body.angular_velocity = 1.0

        def delete_cell(self, cell):
            self.cells.remove(cell)

    class _BoundaryProjectionSimulator:
        def __init__(self, **kwargs):
            self.space = {"dummy_space": True}
            self.colony = _BoundaryProjectionColony()
            self._dt = float(kwargs["physics_config"].DT)
            captured["cell"] = self.colony.cells[0]

        def step(self):
            for cell in self.colony.cells:
                for segment in cell.physics_representation.segments:
                    body = segment.body
                    body.position = body.position + body.velocity * self._dt
                    body.angle = float(body.angle) + float(body.angular_velocity) * self._dt

    monkeypatch.setattr(physics_simulator_module, "Simulator", _BoundaryProjectionSimulator)
    monkeypatch.setattr(cell_snapshot_module, "CellSnapshot", _DummySnapshot)

    simulation = Simulation(
        **kwargs,
        brownian_application_mode="velocity",
        brownian_projection_angular_damping=0.2,
    )
    simulation.run_simulation(show_window=False)

    body = captured["cell"].physics_representation.segments[0].body
    max_x = layout.to_world_point((geometry.inner_half_width - 0.5, 0.0))[0]
    assert float(body.position[0]) <= max_x + 1e-6
    assert float(body.velocity[0]) == pytest.approx(0.0)
    assert abs(float(body.angular_velocity)) < 1.0


def test_run_simulation_brownian_projection_does_not_cap_open_end(tmp_path, monkeypatch):
    captured = {}
    kwargs = _simulation_kwargs(tmp_path)
    geometry, layout = _default_geometry_layout(kwargs)
    start_x, start_y = layout.to_world_point((0.0, geometry.open_end_y - 0.1))
    open_end_y = layout.to_world_point((0.0, geometry.open_end_y))[1]

    class _OpenEndProjectionColony:
        def __init__(self):
            self.cells = [_DummyCellWithPhysics(1, x=start_x, y=start_y)]
            self.cells[0].physics_representation.segments[0].body.velocity = Vec2d(0.0, 20.0)

        def delete_cell(self, cell):
            self.cells.remove(cell)

    class _OpenEndProjectionSimulator:
        def __init__(self, **kwargs):
            self.space = {"dummy_space": True}
            self.colony = _OpenEndProjectionColony()
            self._dt = float(kwargs["physics_config"].DT)
            captured["cell"] = self.colony.cells[0]

        def step(self):
            for cell in self.colony.cells:
                for segment in cell.physics_representation.segments:
                    body = segment.body
                    body.position = body.position + body.velocity * self._dt

    monkeypatch.setattr(physics_simulator_module, "Simulator", _OpenEndProjectionSimulator)
    monkeypatch.setattr(cell_snapshot_module, "CellSnapshot", _DummySnapshot)

    simulation = Simulation(
        **kwargs,
        brownian_application_mode="velocity",
    )
    simulation.run_simulation(show_window=False)

    body = captured["cell"].physics_representation.segments[0].body
    assert float(body.position[1]) > open_end_y
