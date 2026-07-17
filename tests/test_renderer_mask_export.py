from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import SyMBac.renderer as renderer_module
from SyMBac.renderer import Renderer


def _renderer_returning(mask, superres_mask):
    renderer = Renderer.__new__(Renderer)
    renderer.additional_real_images = []
    renderer.params = SimpleNamespace(
        kwargs={
            "noise_var": 0.0,
            "defocus": 0.0,
            "halo_top_intensity": 0.0,
            "halo_bottom_intensity": 0.0,
            "halo_start": 0.0,
            "halo_end": 0.0,
            "cell_texture_strength": 0.0,
            "cell_texture_scale": 1.0,
        }
    )

    def generate_test_comparison(self, **kwargs):
        image = np.arange(mask.size, dtype=float).reshape(mask.shape)
        return image, mask.copy(), superres_mask.copy()

    renderer.generate_test_comparison = MethodType(
        generate_test_comparison,
        renderer,
    )
    return renderer


def _render_parameters():
    return {
        "n_samples": 1,
        "media_multipliers": [1.0],
        "cell_multipliers": [1.0],
        "device_multipliers": [1.0],
        "sigmas": [1.0],
        "scene_nos": [0],
        "hist_match_bools": [False],
        "noise_match_bools": [False],
        "fourier_match_bools": [False],
    }


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        (np.array([[False, True]], dtype=bool), True),
        (np.array([[0, 1]], dtype=np.uint16), True),
        (np.array([[0, 300]], dtype=np.uint16), False),
    ],
)
def test_mask_is_binary_only_for_boolean_or_zero_one_labels(mask, expected):
    assert renderer_module._is_binary_mask(mask) is expected


def test_generate_training_data_relabels_instances_before_uint8_export(tmp_path):
    mask = np.array(
        [
            [0, 255, 256, 512],
            [768, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    superres_mask = np.array(
        [
            [768, 512, 256, 255],
            [0, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    renderer = _renderer_returning(mask, superres_mask)

    with pytest.warns(UserWarning, match="render_sample_parameters"):
        renderer.generate_training_data(
            sample_amount=0,
            randomise_hist_match=False,
            randomise_noise_match=False,
            burn_in=0,
            n_samples=1,
            save_dir=str(tmp_path),
            n_jobs=1,
            mask_dtype=np.uint8,
            render_sample_parameters=_render_parameters(),
        )

    saved_mask = np.asarray(Image.open(tmp_path / "masks" / "Nonesynth_00000.png"))
    saved_superres_mask = np.asarray(
        Image.open(tmp_path / "superres_masks" / "Nonesynth_00000.png")
    )

    assert np.array_equal(
        saved_mask,
        np.array([[0, 1, 2, 3], [4, 0, 0, 0]], dtype=np.uint8),
    )
    assert np.array_equal(
        saved_superres_mask,
        np.array([[4, 3, 2, 1], [0, 0, 0, 0]], dtype=np.uint8),
    )


def test_generate_training_data_rejects_more_instances_than_dtype_can_store(tmp_path):
    mask = np.arange(1, 257, dtype=np.uint16).reshape(16, 16)
    renderer = _renderer_returning(mask, mask)

    with pytest.warns(UserWarning, match="render_sample_parameters"):
        with pytest.raises(ValueError, match="256 instances.*uint8"):
            renderer.generate_training_data(
                sample_amount=0,
                randomise_hist_match=False,
                randomise_noise_match=False,
                burn_in=0,
                n_samples=1,
                save_dir=str(tmp_path),
                n_jobs=1,
                mask_dtype=np.uint8,
                render_sample_parameters=_render_parameters(),
            )
