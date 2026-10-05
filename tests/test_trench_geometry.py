import pymunk
import pytest

from SyMBac.physics.microfluidic_geometry import GeometryLayout, TrenchGeometrySpec
from SyMBac.trench_geometry import get_trench_segments, trench_creator


@pytest.mark.parametrize("trench_length", [4.0, 10.0])
def test_get_trench_segments_finds_walls_in_short_trenches(trench_length):
    geometry = TrenchGeometrySpec(width=18.0, trench_length=trench_length, barrier_thickness=10.0)
    layout = GeometryLayout(geometry, min_preview_size=0)
    space = pymunk.Space()
    geometry.build(space, layout)

    unrelated_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    unrelated_shape = pymunk.Segment(unrelated_body, (-100.0, 0.0), (-100.0, 100.0), 20.0)
    space.add(unrelated_body, unrelated_shape)

    walls = get_trench_segments(space)

    assert len(walls) == 2
    wall_lengths = sorted(abs(float(row.b.y - row.a.y)) for row in walls.itertuples())
    assert wall_lengths == pytest.approx([trench_length, trench_length])


def test_get_trench_segments_uses_each_shapes_body():
    space = pymunk.Space()
    space.add(pymunk.Body(1.0, 1.0))

    for x in (0.0, 10.0):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Segment(body, (x, 0.0), (x, 4.0), 1.0)
        shape.geometry_role = "trench_wall"
        space.add(body, shape)

    walls = get_trench_segments(space)

    assert len(walls) == 2


def test_legacy_trench_creator_marks_short_trench_walls():
    space = pymunk.Space()
    trench_creator(size=18.0, trench_length=4.0, global_xy=(0.0, 0.0), space=space)

    walls = get_trench_segments(space)

    assert len(walls) == 2
    wall_lengths = sorted(abs(float(row.b.y - row.a.y)) for row in walls.itertuples())
    assert wall_lengths == pytest.approx([4.0, 4.0])
