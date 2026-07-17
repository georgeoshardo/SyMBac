import numpy as np
import pytest

from SyMBac.drawing import crop_image, get_crop_bounds_2D, make_images_same_shape


@pytest.mark.parametrize(
    ("region", "expected_shape"),
    [
        ((slice(2, 3), slice(4, 5)), (1, 1)),
        ((slice(1, 5), slice(2, 6)), (4, 4)),
    ],
)
def test_crop_bounds_use_exclusive_stop_indices(region, expected_shape):
    image = np.zeros((7, 8), dtype=np.uint8)
    image[region] = 1

    rows, cols = get_crop_bounds_2D(image)
    cropped = crop_image(image, rows, cols, pad=0)

    assert cropped.shape == expected_shape
    assert np.all(cropped == 1)


@pytest.mark.parametrize(("real_width", "synthetic_width"), [(4, 8), (5, 9)])
def test_make_images_same_shape_centers_same_parity_widths(real_width, synthetic_width):
    real_image = np.zeros((2, real_width))
    synthetic_image = np.tile(np.arange(synthetic_width), (4, 1))

    _, cropped = make_images_same_shape(real_image, synthetic_image, rescale_int=False)

    expected_start = (synthetic_width - real_width) // 2
    expected = synthetic_image[2:, expected_start : expected_start + real_width]
    np.testing.assert_array_equal(cropped, expected)
