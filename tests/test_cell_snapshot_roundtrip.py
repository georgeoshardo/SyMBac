import numpy as np
import pymunk

from SyMBac.cell_snapshot import CellSnapshot
from SyMBac.physics.config import CellConfig
from SyMBac.physics.simcell import SimCell


def test_cell_snapshot_round_trip_from_real_simcell():
    space = pymunk.Space()
    cell = SimCell(space=space, config=CellConfig(), start_pos=(1, 2), group_id=3)

    snapshot = CellSnapshot(
        simcell=cell,
        t=5,
        mother_mask_label=1,
        generation=2,
        just_divided=True,
    )

    segment_dict = snapshot.to_segment_dict()

    assert segment_dict["mask_label"] == 3
    assert segment_dict["cell_id"] == 3
    assert segment_dict["positions"].shape[0] == len(cell.physics_representation.segments)
    assert segment_dict["radii"].shape[0] == len(cell.physics_representation.segments)
    assert isinstance(snapshot.position, pymunk.Vec2d)
    assert snapshot.t == 5
    assert snapshot.generation == 2
    assert snapshot.mother_mask_label == 1
    assert snapshot.just_divided is True
    assert np.all(segment_dict["radii"] > 0)
