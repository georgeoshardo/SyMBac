import pytest
import pymunk

from SyMBac.physics.config import CellConfig
from SyMBac.physics.division_manager import DivisionManager
from SyMBac.physics.simcell import SimCell


def test_simcell_birth_length_division_site_and_group_id_behaviour():
    space = pymunk.Space()
    cell = SimCell(space=space, config=CellConfig(), start_pos=(0, 0), group_id=42)

    assert cell.birth_length > 0

    cell.division_site = 1
    assert cell.division_site == 1

    cell.division_site = None
    assert cell.division_site == len(cell.physics_representation.segments) // 2

    assert cell.group_id == 42
    with pytest.raises(AttributeError, match="immutable"):
        cell.group_id = 7


def test_seed_cells_sample_max_length_above_birth_length():
    space = pymunk.Space()
    config = CellConfig(BASE_MAX_LENGTH=1.0, SEED_CELL_SEGMENTS=40)
    cell = SimCell(space=space, config=config, start_pos=(0, 0), group_id=1)

    assert cell.max_length > cell.birth_length
    assert DivisionManager(space=space, config=config).ready_to_divide(cell) is False
