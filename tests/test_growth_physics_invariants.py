import pymunk
import pytest

import SyMBac.simulation as simulation_module
from SyMBac.physics.config import CellConfig, PhysicsConfig
from SyMBac.physics.division_manager import DivisionManager
from SyMBac.physics.growth_manager import GrowthManager
from SyMBac.physics.segments import CellSegment
from SyMBac.physics.simcell import SimCell
from SyMBac.physics.simulator import Simulator


def test_continuous_length_cell_can_grow_to_division_segment_count(monkeypatch):
    config = CellConfig(
        GRANULARITY=4,
        SEGMENT_RADIUS=4.0,
        GROWTH_RATE=1.0,
        MIN_LENGTH_AFTER_DIVISION=3,
        BASE_MAX_LENGTH=6.0,
        SEED_CELL_SEGMENTS=2,
        NOISE_STRENGTH=0.0,
        SIMPLE_LENGTH=False,
    )
    space = pymunk.Space()
    cell = SimCell(space=space, config=config, start_pos=(0, 0))
    division_manager = DivisionManager(space=space, config=config)
    monkeypatch.setattr(
        "SyMBac.physics.growth_manager.np.random.uniform",
        lambda *_args: 2.0,
    )

    assert cell.length >= cell.max_length
    assert not division_manager.ready_to_divide(cell)

    GrowthManager.grow(cell, dt=1.0)
    GrowthManager.grow(cell, dt=1.0)

    assert cell.physics_representation.num_segments == 6
    assert division_manager.ready_to_divide(cell)


def test_growth_preserves_overshoot_and_inserts_every_crossed_threshold(monkeypatch):
    config = CellConfig(
        GRANULARITY=4,
        SEGMENT_RADIUS=4.0,
        GROWTH_RATE=1.0,
        MIN_LENGTH_AFTER_DIVISION=2,
        BASE_MAX_LENGTH=100.0,
        SEED_CELL_SEGMENTS=3,
        NOISE_STRENGTH=0.0,
    )
    cell = SimCell(space=pymunk.Space(), config=config, start_pos=(0, 0))
    monkeypatch.setattr(
        "SyMBac.physics.growth_manager.np.random.uniform",
        lambda *_args: 7.0,
    )

    GrowthManager.grow(cell, dt=1.0)

    representation = cell.physics_representation
    assert representation.num_segments == 9
    assert representation.growth_accumulator_head == pytest.approx(0.5)
    assert representation.growth_accumulator_tail == pytest.approx(0.5)
    assert representation.pivot_joints[0].anchor_a.x == pytest.approx(1.0)
    assert representation.pivot_joints[-1].anchor_b.x == pytest.approx(-1.0)


def test_radius_update_refreshes_moment_and_spatial_index():
    config = CellConfig(SEGMENT_RADIUS=2.0)
    space = pymunk.Space()
    segment = CellSegment(
        config=config,
        group_id=1,
        position=pymunk.Vec2d(0, 0),
        space=space,
    )
    space.add(segment.body, segment.shape)

    assert space.point_query_nearest((3.0, 0.0), 0.0, pymunk.ShapeFilter()) is None

    segment.radius = 4.0

    expected_moment = pymunk.moment_for_circle(segment.body.mass, 0.0, 4.0)
    assert segment.body.moment == pytest.approx(expected_moment)
    hit = space.point_query_nearest((3.0, 0.0), 0.0, pymunk.ShapeFilter())
    assert hit is not None
    assert hit.shape is segment.shape


def test_adaptive_iterations_decay_to_configured_baseline():
    simulator = Simulator(
        physics_config=PhysicsConfig(ITERATIONS=23),
        initial_cell_config=CellConfig(GROWTH_RATE=0.0, NOISE_STRENGTH=0.0),
        adaptive_iterations=True,
    )
    simulator.space.iterations = 24

    simulator.step()

    assert simulator.space.iterations == 23


def test_lysis_randomizes_candidates_before_preserving_one_survivor(monkeypatch):
    handle_lysis = getattr(simulation_module, "_handle_lysis", None)
    assert callable(handle_lysis)

    cells = [object(), object(), object()]

    class _Colony:
        def __init__(self):
            self.cells = cells.copy()
            self.deleted = []

        def delete_cell(self, cell):
            self.deleted.append(cell)
            self.cells.remove(cell)

    class _Simulator:
        def __init__(self):
            self.colony = _Colony()

    simulator = _Simulator()
    monkeypatch.setattr(simulation_module.np.random, "shuffle", lambda values: values.reverse())
    monkeypatch.setattr(simulation_module.norm, "rvs", lambda: 0.0)

    handle_lysis(simulator, lysis_p=1.0)

    assert simulator.colony.deleted == [cells[2], cells[1]]
    assert simulator.colony.cells == [cells[0]]
