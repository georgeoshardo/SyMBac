import pytest
import pymunk

from SyMBac.physics.config import CellConfig
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
