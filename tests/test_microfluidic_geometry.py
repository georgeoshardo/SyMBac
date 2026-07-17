import numpy as np
import pymunk
import pytest

from SyMBac.physics.microfluidic_geometry import GeometryLayout, TrenchGeometrySpec


@pytest.fixture
def trench_geometry():
    geometry = TrenchGeometrySpec(width=18.0, trench_length=39.0, barrier_thickness=10.0)
    return geometry, GeometryLayout(geometry, min_preview_size=0)


def test_geometry_layout_offsets_trench_into_positive_padded_world_bounds():
    geometry = TrenchGeometrySpec(width=18.0, trench_length=39.0, barrier_thickness=10.0)
    layout = GeometryLayout(geometry, min_preview_size=0)

    assert layout.world_bounds.min_x == pytest.approx(layout.padding_x)
    assert layout.world_bounds.min_y == pytest.approx(layout.padding_y)
    assert layout.world_bounds.max_x > layout.world_bounds.min_x
    assert layout.world_bounds.max_y > layout.world_bounds.min_y
    assert layout.preview_shape[0] >= int(layout.world_bounds.max_y)
    assert layout.preview_shape[1] >= int(layout.world_bounds.max_x)


@pytest.mark.parametrize(
    ("local_position", "expected"),
    [
        ((0.0, 1.9), False),
        ((0.0, 2.0), True),
        ((6.0, 5.0), False),
        ((6.0, 5.5), True),
    ],
)
def test_trench_containment_follows_rounded_cap(trench_geometry, local_position, expected):
    geometry, layout = trench_geometry
    positions = np.array([layout.to_world_point(local_position)])

    result = geometry.positions_within_bounds(
        positions,
        radii=[2.0],
        layout=layout,
        enforce_open_end_cap=False,
    )

    assert result is expected


def test_trench_projection_moves_cells_clear_of_rounded_cap(trench_geometry):
    geometry, layout = trench_geometry
    body = pymunk.Body(1.0, 1.0)
    body.position = layout.to_world_point((6.0, 5.0))
    body.velocity = (1.0, -2.0)

    projected, changed_x, changed_y = geometry.project_body_inside_bounds(body, radius=2.0, layout=layout)

    expected_y = geometry.inner_half_width - np.sqrt((geometry.inner_half_width - 2.0) ** 2 - 6.0**2)
    local_position = layout.to_local_point(body.position)
    assert projected
    assert not changed_x
    assert changed_y
    assert local_position == pytest.approx((6.0, expected_y))
    assert body.velocity == pytest.approx((1.0, 0.0))
