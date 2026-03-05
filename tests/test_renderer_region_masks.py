import numpy as np

from SyMBac.renderer import Renderer


class _Layer:
    def __init__(self, data):
        self.data = data


def _build_renderer_stub(real_resize):
    renderer = Renderer.__new__(Renderer)
    renderer.real_resize = np.asarray(real_resize, dtype=np.float32)
    renderer.image_params = None
    renderer._explicit_region_masks = None
    return renderer


def test_set_region_masks_overrides_auto_segmentation():
    real = np.array(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=np.float32,
    )
    renderer = _build_renderer_stub(real)
    renderer.auto_segment_regions = lambda *args, **kwargs: {
        "media": np.zeros_like(real, dtype=bool),
        "cell": np.zeros_like(real, dtype=bool),
        "device": np.ones_like(real, dtype=bool),
    }

    renderer.set_region_masks(
        media_mask=np.array([[1, 1], [0, 0]], dtype=np.uint8),
        cell_mask=np.array([[0, 0], [1, 0]], dtype=np.uint8),
        device_mask=np.array([[0, 0], [0, 1]], dtype=np.uint8),
    )
    renderer._ensure_image_params()

    media_mean, cell_mean, device_mean, *_ = renderer.image_params
    assert media_mean == np.mean([1.0, 2.0])
    assert cell_mean == 3.0
    assert device_mean == 4.0


def test_clear_region_masks_falls_back_to_layer_data_then_auto():
    real = np.array(
        [[10.0, 20.0], [30.0, 40.0]],
        dtype=np.float32,
    )
    renderer = _build_renderer_stub(real)
    renderer.auto_segment_regions = lambda *args, **kwargs: {
        "media": np.array([[0, 0], [1, 0]], dtype=bool),
        "cell": np.array([[1, 0], [0, 0]], dtype=bool),
        "device": np.array([[0, 1], [0, 1]], dtype=bool),
    }

    renderer.set_region_masks(
        media_mask=np.array([[1, 0], [0, 0]], dtype=np.uint8),
        cell_mask=np.array([[0, 1], [0, 0]], dtype=np.uint8),
        device_mask=np.array([[0, 0], [1, 1]], dtype=np.uint8),
    )
    renderer.clear_region_masks()

    renderer.media_label = _Layer(np.array([[0, 1], [0, 0]], dtype=np.uint8))
    renderer.cell_label = _Layer(np.array([[1, 0], [0, 0]], dtype=np.uint8))
    renderer.device_label = _Layer(np.array([[0, 0], [1, 1]], dtype=np.uint8))

    renderer._ensure_image_params()
    media_mean, cell_mean, device_mean, *_ = renderer.image_params
    assert media_mean == 20.0
    assert cell_mean == 10.0
    assert device_mean == np.mean([30.0, 40.0])


def test_set_region_masks_requires_matching_shape():
    renderer = _build_renderer_stub(np.zeros((4, 4), dtype=np.float32))

    try:
        renderer.set_region_masks(
            media_mask=np.zeros((3, 4), dtype=bool),
            cell_mask=np.zeros((4, 4), dtype=bool),
            device_mask=np.zeros((4, 4), dtype=bool),
        )
    except ValueError as exc:
        assert "must match real image shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched mask shapes.")
