import numpy as np

from SyMBac.live_viewer import render_live_frame_image


def test_render_live_frame_image_draws_trench_and_cells():
    frame_segments = [
        (
            np.array([[40.0, 12.0], [40.0, 20.0], [40.0, 28.0]], dtype=np.float64),
            np.array([3.0, 3.0, 3.0], dtype=np.float64),
        )
    ]

    image = render_live_frame_image(
        frame_segments,
        scene_shape=(64, 96),
        trench_center_x=35.0,
        trench_width=18.0,
        trench_height=48.0,
    )

    assert image.shape == (64, 96)
    assert image.dtype == np.uint8
    assert image.max() == 255
    assert np.count_nonzero(image == 255) > 0
    assert np.count_nonzero(image == 96) > 0
    assert image[9, 26] == 96
    assert image[9, 44] == 96
