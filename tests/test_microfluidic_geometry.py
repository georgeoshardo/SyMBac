import pytest

from SyMBac.physics.microfluidic_geometry import GeometryLayout, TrenchGeometrySpec


def test_geometry_layout_offsets_trench_into_positive_padded_world_bounds():
    geometry = TrenchGeometrySpec(width=18.0, trench_length=39.0, barrier_thickness=10.0)
    layout = GeometryLayout(geometry, min_preview_size=0)

    assert layout.world_bounds.min_x == pytest.approx(layout.padding_x)
    assert layout.world_bounds.min_y == pytest.approx(layout.padding_y)
    assert layout.world_bounds.max_x > layout.world_bounds.min_x
    assert layout.world_bounds.max_y > layout.world_bounds.min_y
    assert layout.preview_shape[0] >= int(layout.world_bounds.max_y)
    assert layout.preview_shape[1] >= int(layout.world_bounds.max_x)
