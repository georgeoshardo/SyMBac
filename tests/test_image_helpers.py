import numpy as np

from SyMBac.misc import resize_mask, unet_weight_map
from SyMBac.pySHINE import lumMatch, sfMatch


def test_lum_match_does_not_average_explicit_luminance_over_images():
    images = [
        np.array([[0.0, 1.0], [2.0, 3.0]]),
        np.array([[4.0, 6.0], [8.0, 10.0]]),
    ]

    matched = lumMatch(images, lum=(10, 2))

    for image in matched:
        np.testing.assert_allclose((np.mean(image), np.std(image)), (10, 2))


def test_lum_match_infers_masked_mean_and_standard_deviation():
    images = [
        np.array([[0.0, 2.0], [0.0, 2.0]]),
        np.array([[10.0, 14.0], [10.0, 14.0]]),
    ]
    masks = [np.ones((2, 2), dtype=bool) for _ in images]

    matched = lumMatch(images, mask=masks)

    for image in matched:
        np.testing.assert_allclose((np.mean(image), np.std(image)), (6.5, 1.5))


def test_sf_match_returns_finite_images_for_zero_radial_energy():
    images = [np.zeros((8, 8)), np.full((8, 8), 42.0)]

    matched = sfMatch(images)

    assert all(np.isfinite(image).all() for image in matched)


def test_sf_match_preserves_existing_nondegenerate_result():
    images = [
        np.array(
            [
                [207.0, 22.0, 46.0, 61.0],
                [47.0, 205.0, 222.0, 149.0],
                [11.0, 25.0, 85.0, 111.0],
                [159.0, 123.0, 68.0, 41.0],
            ]
        ),
        np.array(
            [
                [177.0, 188.0, 9.0, 29.0],
                [116.0, 100.0, 227.0, 132.0],
                [108.0, 110.0, 170.0, 150.0],
                [45.0, 189.0, 193.0, 244.0],
            ]
        ),
    ]
    expected = np.array(
        [
            [
                [215.92342446, 45.20166264, 55.74891053, 88.05443828],
                [76.06261371, 217.33796077, 249.89337633, 157.90259746],
                [20.89496420, 51.96898037, 89.95033884, 140.66773372],
                [188.01957864, 134.64438587, 102.38161631, 49.84741786],
            ],
            [
                [153.57458860, 173.64724241, -12.36475379, 14.15291499],
                [100.71720676, 78.32035547, 208.18471222, 107.53532694],
                [81.93388727, 94.49215378, 151.93353598, 139.13634474],
                [33.24825836, 164.25409447, 178.67116675, 217.06296505],
            ],
        ]
    )

    matched = sfMatch(images)

    np.testing.assert_allclose(matched, expected, rtol=1e-8, atol=1e-8)


def test_resize_mask_preserves_touching_integer_labels():
    mask = np.array([[1, 2], [1, 2]], dtype=np.uint16)

    resized = resize_mask(mask, (4, 4), ret_label=True)

    expected = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(resized, expected)


def test_unet_weight_map_preserves_fractional_class_weights():
    mask = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.uint8)

    weights = unet_weight_map(mask, wc={0: 0.25, 1: 1.5})

    assert np.issubdtype(weights.dtype, np.floating)
    np.testing.assert_array_equal(weights, np.where(mask == 0, 0.25, 1.5))
