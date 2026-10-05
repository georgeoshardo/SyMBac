"""The hot-path optimisations in the physics layer must not change results."""
import math

import numpy as np
import pymunk

from SyMBac.physics.config import CellConfig, PhysicsConfig
from SyMBac.physics.simcell import SimCell
from SyMBac.physics.simulator import Simulator


def _make_cell(noise=0.05, **kwargs):
    space = pymunk.Space()
    config = CellConfig(NOISE_STRENGTH=noise, **kwargs)
    return space, SimCell(space=space, config=config, start_pos=(0, 0), group_id=1)


def test_continuous_length_matches_vec2d_reference():
    _, cell = _make_cell()
    pr = cell.physics_representation
    for i, segment in enumerate(pr.segments):  # bend the chain so the sum is non-trivial
        segment.body.position = (i * 1.9, 0.3 * math.sin(i))

    reference = 0.0
    for a, b in zip(pr.segments, pr.segments[1:]):
        reference += (b.position - a.position).length
    reference += pr.segments[0].radius + pr.segments[-1].radius

    assert pr.get_continuous_length() == reference  # bit-identical, not approx


def test_noise_draws_the_same_random_stream_as_the_scalar_formulation():
    space, cell = _make_cell(noise=0.05)
    pr = cell.physics_representation
    n = len(pr.segments)
    strength = cell.config.NOISE_STRENGTH

    np.random.seed(123)
    expected = []
    for _ in range(n):
        fx = np.random.uniform(-strength, strength)
        fy = np.random.uniform(-strength, strength)
        torque = np.random.uniform(-strength * 0.1, strength * 0.1)
        expected.append((fx, fy, torque))

    np.random.seed(123)
    pr.apply_noise(1 / 60)

    for segment, (fx, fy, torque) in zip(pr.segments, expected):
        assert tuple(segment.body.force) == (fx, fy)
        assert segment.body.torque == torque


def test_noise_accumulates_on_top_of_existing_force():
    _, cell = _make_cell(noise=0.05)
    body = cell.physics_representation.segments[0].body
    body.force = (10.0, -4.0)
    body.torque = 2.0
    cell.physics_representation.apply_noise(1 / 60)
    assert abs(body.force.x - 10.0) <= 0.05
    assert abs(body.force.y + 4.0) <= 0.05
    assert abs(body.torque - 2.0) <= 0.005


def test_segment_radius_cache_tracks_the_shape():
    _, cell = _make_cell()
    segment = cell.physics_representation.segments[0]
    segment.radius = segment.radius  # no-op path
    assert segment.radius == segment.shape.radius

    segment.radius = 3.25
    assert segment.radius == 3.25 == segment.shape.radius

    segment.radius = 3.25  # repeat is a no-op, state stays consistent
    assert segment.shape.radius == 3.25

    segment.radius = 7.5
    assert segment.shape.radius == 7.5


def test_step_cache_is_off_outside_step_and_dropped_when_chain_changes():
    _, cell = _make_cell()
    pr = cell.physics_representation

    pr.begin_step_cache()
    first = pr.get_continuous_length()
    pr.segments[-1].body.position = pr.segments[-1].body.position + (5.0, 0.0)
    assert pr.get_continuous_length() == first  # memo is deliberately reused inside the block
    pr.end_step_cache()

    assert pr.get_continuous_length() > first  # live read once the block has ended

    for op in (pr.add_tail_segment, pr.add_head_segment, pr.remove_tail_segment, pr.remove_head_segment):
        pr.begin_step_cache()
        pr.get_continuous_length()
        assert pr._chain_length_memo is not None
        op()  # chain changes -> memo must be dropped
        assert pr._chain_length_memo is None
        cached = pr.get_continuous_length()
        pr.end_step_cache()
        assert cached == pr.get_continuous_length()  # memo rebuilt from live positions


def test_simulator_leaves_step_cache_disabled_between_steps():
    sim = Simulator(
        PhysicsConfig(ITERATIONS=10),
        CellConfig(SIMPLE_LENGTH=False, SEED_CELL_SEGMENTS=6, MIN_LENGTH_AFTER_DIVISION=3),
    )
    for _ in range(5):
        sim.step()
    assert all(not c.physics_representation._step_cache_active for c in sim.cells)


def test_seeded_colony_growth_is_reproducible():
    def run():
        np.random.seed(5)
        sim = Simulator(
            PhysicsConfig(ITERATIONS=20),
            CellConfig(SIMPLE_LENGTH=False, SEED_CELL_SEGMENTS=6, MIN_LENGTH_AFTER_DIVISION=3,
                       BASE_MAX_LENGTH=25, GROWTH_RATE=40.0, SEGMENT_RADIUS=5.0, GRANULARITY=4),
        )
        for _ in range(400):
            sim.step()
        return sim.num_cells, [tuple(s.position) for c in sim.cells for s in c.physics_representation.segments]

    a, b = run(), run()
    assert a == b
    assert a[0] > 1  # divisions actually happened
