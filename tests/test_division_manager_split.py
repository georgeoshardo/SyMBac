import pytest
import pymunk

from SyMBac.physics.config import CellConfig
from SyMBac.physics.division_manager import DivisionManager
from SyMBac.physics.simcell import SimCell


def test_split_cell_resets_both_growth_accumulators_and_birth_lengths():
    space = pymunk.Space()
    config = CellConfig(SEED_CELL_SEGMENTS=12, MIN_LENGTH_AFTER_DIVISION=3)
    manager = DivisionManager(space=space, config=config)
    mother = SimCell(space=space, config=config, start_pos=(0, 0), group_id=1)

    mother.num_segments_at_division_start = mother.physics_representation.num_segments
    mother.division_bias = 0
    mother.physics_representation.growth_accumulator_head = 2.5
    mother.physics_representation.growth_accumulator_tail = 3.5

    daughter = manager.split_cell(mother, next_group_id=2)

    assert mother.physics_representation.growth_accumulator_head == pytest.approx(0.0)
    assert mother.physics_representation.growth_accumulator_tail == pytest.approx(0.0)
    assert daughter.physics_representation.growth_accumulator_head == pytest.approx(0.0)
    assert daughter.physics_representation.growth_accumulator_tail == pytest.approx(0.0)

    assert mother.birth_length == pytest.approx(mother.length)
    assert daughter.birth_length == pytest.approx(daughter.length)


def test_septum_radius_updates_keep_body_moment_in_sync():
    space = pymunk.Space()
    config = CellConfig(SEED_CELL_SEGMENTS=12, MIN_LENGTH_AFTER_DIVISION=3)
    manager = DivisionManager(space=space, config=config)
    cell = SimCell(space=space, config=config, start_pos=(0, 0), group_id=1)

    manager.set_division_readiness(cell)
    manager.initialise_mother_daughter_septum_segments(cell)
    cell.septum_progress = 0.5

    mother_segment = cell.physics_representation._mother_septum_segments[0]
    original_moment = mother_segment.body.moment

    manager.update_septum_segment_radii(cell)

    assert mother_segment.radius < cell.current_segment_radius
    assert mother_segment.body.moment != pytest.approx(original_moment)
