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
