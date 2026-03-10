import numpy as np

from SyMBac.live_viewer import render_live_frame_image
from SyMBac.physics.microfluidic_geometry import GeometryLayout, TrenchGeometrySpec


def test_render_live_frame_image_draws_trench_and_cells():
    geometry = TrenchGeometrySpec(width=18.0, trench_length=39.0)
    layout = GeometryLayout(geometry)
    static_segments = geometry.preview_primitives(layout)
    frame_segments = [
        (
            np.array(
                [
                    layout.to_world_point((0.0, 12.0)),
                    layout.to_world_point((0.0, 20.0)),
                    layout.to_world_point((0.0, 28.0)),
                ],
                dtype=np.float64,
            ),
            np.array([3.0, 3.0, 3.0], dtype=np.float64),
        )
    ]

    image = render_live_frame_image(
        frame_segments,
        static_segments,
        layout.preview_shape,
    )
    left_wall = static_segments[-2]
    right_wall = static_segments[-1]
    wall_y = int(round((left_wall.p1[1] + left_wall.p2[1]) / 2.0))
    left_x = int(round(left_wall.p1[0]))
    right_x = int(round(right_wall.p1[0]))

    assert image.shape == layout.preview_shape
    assert image.dtype == np.uint8
    assert image.max() == 255
    assert np.count_nonzero(image == 255) > 0
    assert np.count_nonzero(image == 96) > 0
    assert image[wall_y, left_x] == 96
    assert image[wall_y, right_x] == 96
