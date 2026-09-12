import random
import threading
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
            "media_multiplier": 10.0,
            "cell_multiplier": 2.0,
            "device_multiplier": 5.0,
            "sigma": 1.5,
            "match_histogram": False,
            "match_noise": False,
            "match_fourier": False,
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


def _comparison_renderer(mode="simple fluo", camera=None):
    renderer = Renderer.__new__(Renderer)
    scene = np.arange(1, 65, dtype=float).reshape(8, 8)
    mask = np.zeros((8, 8), dtype=np.uint16)
    mask[2:6, 2:6] = 1

    renderer.simulation = SimpleNamespace(
        OPL_scenes=[scene],
        masks=[mask],
        resize_amount=1,
        pix_mic_conv=1.0,
    )
    renderer.real_image = np.arange(1, 17, dtype=float).reshape(4, 4)
    renderer.PSF = SimpleNamespace(
        mode=mode,
        R=1.0,
        W=1.0,
        radius=1,
        scale=1.0,
        NA=0.8,
        n=1.33,
        apo_sigma=1.0,
        wavelength=0.6,
        condenser="Ph1",
        kernel=np.array([[1.0]]),
    )
    renderer.camera = camera
    renderer.additional_real_images = []
    renderer.x_border_expansion_coefficient = 1
    renderer.y_border_expansion_coefficient = 1
    renderer.image_params = (0.5, 0.5, 0.5, None, 0.1, 0.1, 0.1, None)
    renderer.error_params = tuple([] for _ in range(8))
    renderer.params = SimpleNamespace(
        kwargs={
            "noise_var": 0.01,
            "defocus": 0.0,
            "halo_top_intensity": 1.0,
            "halo_bottom_intensity": 1.0,
            "halo_start": 0.0,
            "halo_end": 1.0,
            "cell_texture_strength": 0.0,
            "cell_texture_scale": 1.0,
        }
    )

    def generate_PC_OPL(
        self,
        scene,
        mask,
        media_multiplier,
        cell_multiplier,
        device_multiplier,
        **_kwargs,
    ):
        expanded_scene = np.array(scene, copy=True)
        expanded_scene[0, 0] = media_multiplier
        expanded_scene_no_cells = np.zeros_like(expanded_scene)
        expanded_scene_no_cells[2:6, 2:6] = device_multiplier
        return expanded_scene, expanded_scene_no_cells, np.array(mask, copy=True)

    renderer.generate_PC_OPL = MethodType(generate_PC_OPL, renderer)
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


def test_parallel_training_uses_each_samples_psf_kernel(tmp_path, monkeypatch):
    renderer = _comparison_renderer(mode="phase contrast")
    original_psf = renderer.PSF
    psf_ready = threading.Barrier(2, timeout=5)
    convolutions = {}

    class SynchronizedPSF:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.R = 1.0
            self.W = 1.0
            self.scale = 1.0

        def calculate_PSF(self):
            self.kernel = np.array([[self.apo_sigma]])
            psf_ready.wait()

    def record_convolution(image, kernel, _rescale_factor, rescale_int):
        convolutions[float(image[0, 0])] = float(kernel[0, 0])
        return np.arange(image.size, dtype=float).reshape(image.shape)

    monkeypatch.setattr(renderer_module, "PSF_generator", SynchronizedPSF)
    monkeypatch.setattr(renderer_module, "convolve_rescale", record_convolution)

    parameters = {
        "n_samples": 2,
        "media_multipliers": [10.0, 20.0],
        "cell_multipliers": [1.0, 1.0],
        "device_multipliers": [2.0, 2.0],
        "sigmas": [1.0, 2.0],
        "scene_nos": [0, 0],
        "hist_match_bools": [False, False],
        "noise_match_bools": [False, False],
        "fourier_match_bools": [False, False],
    }

    with pytest.warns(UserWarning, match="render_sample_parameters"):
        renderer.generate_training_data(
            sample_amount=0,
            randomise_hist_match=False,
            randomise_noise_match=False,
            burn_in=0,
            n_samples=2,
            save_dir=str(tmp_path),
            n_jobs=2,
            render_sample_parameters=parameters,
        )

    assert convolutions == {10.0: 1.0, 20.0: 2.0}
    assert renderer.PSF is original_psf


@pytest.mark.parametrize(
    "camera",
    [None, SimpleNamespace(baseline=1.0, sensitivity=1.0, dark_noise=0.5)],
    ids=["skimage-noise", "camera-noise"],
)
def test_generate_test_comparison_noise_respects_rng(camera):
    renderer = _comparison_renderer(camera=camera)
    render_kwargs = {
        "media_multiplier": 10.0,
        "cell_multiplier": 1.0,
        "device_multiplier": 2.0,
        "sigma": 1.0,
        "scene_no": 0,
        "match_fourier": False,
        "match_histogram": False,
        "match_noise": False,
        "noise_var": 0.01,
        "defocus": 0.0,
        "halo_top_intensity": 1.0,
        "halo_bottom_intensity": 1.0,
        "halo_start": 0.0,
        "halo_end": 1.0,
    }

    first = renderer.generate_test_comparison(
        **render_kwargs,
        rng=np.random.default_rng(0),
    )[0]
    repeated = renderer.generate_test_comparison(
        **render_kwargs,
        rng=np.random.default_rng(0),
    )[0]
    changed = renderer.generate_test_comparison(
        **render_kwargs,
        rng=np.random.default_rng(1),
    )[0]

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, changed)


def test_seed_reproduces_parameters_real_image_selection_and_noise(tmp_path):
    captures = []
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    renderer = _renderer_returning(mask, mask)
    renderer.simulation = SimpleNamespace(sim_length=8)
    renderer.additional_real_images = [
        np.full((2, 2), 1.0),
        np.full((2, 2), 2.0),
    ]

    def generate_test_comparison(self, **kwargs):
        rng = kwargs["rng"]
        captures.append(
            (
                kwargs["media_multiplier"],
                float(kwargs["random_real_image"][0, 0]),
                rng.random(4),
            )
        )
        return rng.random((2, 2)), mask.copy(), mask.copy()

    renderer.generate_test_comparison = MethodType(generate_test_comparison, renderer)

    def render(seed, global_seed, output_dir):
        random.seed(global_seed)
        np.random.seed(global_seed)
        captures.clear()
        renderer.generate_training_data(
            sample_amount=0.2,
            randomise_hist_match=True,
            randomise_noise_match=True,
            burn_in=0,
            n_samples=6,
            save_dir=str(output_dir),
            seed=seed,
            n_jobs=1,
        )
        return [
            (media_multiplier, real_image, noise.copy())
            for media_multiplier, real_image, noise in captures
        ]

    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    try:
        first = render(0, 1, tmp_path / "first")
        repeated = render(0, 5, tmp_path / "repeated")
        changed = render(1, 1, tmp_path / "changed")
    finally:
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)

    for first_sample, repeated_sample in zip(first, repeated):
        assert first_sample[:2] == repeated_sample[:2]
        assert np.array_equal(first_sample[2], repeated_sample[2])
    assert any(
        first_sample[:2] != changed_sample[:2]
        or not np.array_equal(first_sample[2], changed_sample[2])
        for first_sample, changed_sample in zip(first, changed)
    )


def test_timeseries_seed_reproduces_parallel_frame_noise(tmp_path):
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint16)
    renderer = _renderer_returning(mask, mask)
    renderer.simulation = SimpleNamespace(sim_length=4)

    def generate_test_comparison(self, **kwargs):
        return kwargs["rng"].random((2, 2)), mask.copy(), mask.copy()

    renderer.generate_test_comparison = MethodType(generate_test_comparison, renderer)

    for output_dir in (tmp_path / "first", tmp_path / "repeated"):
        renderer.generate_timeseries_training_data(
            save_dir=str(output_dir),
            burn_in=0,
            sample_amount=0.0,
            n_series=1,
            frames_per_series=2,
            export_geff=False,
            seed=0,
            n_jobs=2,
            image_format="png",
        )

    for frame_number in range(2):
        relative_path = f"series_000/images/frame_{frame_number:05d}.png"
        first = np.asarray(Image.open(tmp_path / "first" / relative_path))
        repeated = np.asarray(Image.open(tmp_path / "repeated" / relative_path))
        assert np.array_equal(first, repeated)


def test_repeated_export_advances_number_and_prefixes_zero_cell_masks(tmp_path):
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    renderer = _renderer_returning(mask, mask)

    with pytest.warns(UserWarning, match="render_sample_parameters"):
        renderer.generate_training_data(
            sample_amount=0,
            randomise_hist_match=False,
            randomise_noise_match=False,
            burn_in=0,
            n_samples=1,
            save_dir=str(tmp_path),
            n_jobs=1,
            prefix="batch_",
            render_sample_parameters=_render_parameters(),
        )

    zero_cell_parameters = _render_parameters()
    zero_cell_parameters["cell_multipliers"] = [0.0]
    with pytest.warns(UserWarning, match="render_sample_parameters"):
        renderer.generate_training_data(
            sample_amount=0,
            randomise_hist_match=False,
            randomise_noise_match=False,
            burn_in=0,
            n_samples=1,
            save_dir=str(tmp_path),
            n_jobs=1,
            prefix="batch_",
            render_sample_parameters=zero_cell_parameters,
        )

    expected_names = ["batch_synth_00000.png", "batch_synth_00001.png"]
    for subdirectory in ("convolutions", "masks", "superres_masks"):
        assert sorted(path.name for path in (tmp_path / subdirectory).iterdir()) == expected_names
