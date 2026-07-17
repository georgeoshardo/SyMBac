from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from SyMBac import colony_renderer, colony_simulation
from SyMBac.colony_renderer import ColonyRenderer
from SyMBac.colony_simulation import ColonySimulation
from SyMBac.renderer import convolve_rescale


def _write_png(path, value):
    Image.fromarray(np.full((3, 3), value, dtype=np.uint8)).save(path)


def _simulation_paths(tmp_path):
    paths = {}
    for name in ("masks", "opl", "fluorescence", "fluorescence_3d"):
        paths[name] = tmp_path / name
        paths[name].mkdir()
    return paths


def test_float_images_are_scaled_to_uint16_for_png():
    image = np.array([[-2.0, 0.0, 2.0]], dtype=np.float32)

    converted = colony_renderer._as_png_uint16(image)

    assert converted.dtype == np.uint16
    np.testing.assert_array_equal(converted, [[0, 32768, 65535]])


def test_label_composition_preserves_labels_and_single_pixel_cells():
    mask = np.array([[0, 0], [0, 1]], dtype=np.uint16)
    cell_pixels = np.array([[True, True], [False, False]])

    composed = colony_simulation._compose_label_mask(mask, cell_pixels, 2)

    np.testing.assert_array_equal(composed, [[2, 2], [0, 1]])
    assert set(np.unique(composed)) == {0, 1, 2}


def test_rectangular_scene_uses_independent_axis_offsets():
    x, y = colony_simulation._scene_position(position=(-4, 1), scene_shape=(7, 11))

    assert (x, y) == (2, 5)


def test_simple_fluorescence_reads_fluorescence_scenes(tmp_path):
    paths = _simulation_paths(tmp_path)
    _write_png(paths["opl"] / "opl.png", 1)
    _write_png(paths["fluorescence"] / "fluorescence.png", 2)
    simulation = SimpleNamespace(
        scene_shape=(3, 3),
        resize_amount=1,
        masks_dir=str(paths["masks"]),
        OPL_dir=str(paths["opl"]),
        fluorescence_dir=str(paths["fluorescence"]),
        fluorescent_projections_dir=str(paths["fluorescence_3d"]),
    )

    renderer = ColonyRenderer(
        simulation=simulation,
        PSF=SimpleNamespace(mode="simple fluo"),
    )

    assert renderer.OPL_dirs == [str(paths["fluorescence"] / "fluorescence.png")]


def test_serial_generation_uses_output_indices_and_writes_valid_pngs(
    tmp_path, monkeypatch
):
    paths = _simulation_paths(tmp_path)
    for index in range(2):
        _write_png(paths["opl"] / f"{index}.png", index)
        _write_png(paths["masks"] / f"{index}.png", index + 1)
    simulation = SimpleNamespace(
        scene_shape=(3, 3),
        resize_amount=1,
        masks_dir=str(paths["masks"]),
        OPL_dir=str(paths["opl"]),
        fluorescence_dir=str(paths["fluorescence"]),
        fluorescent_projections_dir=str(paths["fluorescence_3d"]),
    )
    renderer = ColonyRenderer(
        simulation=simulation,
        PSF=SimpleNamespace(mode="phase contrast"),
    )
    monkeypatch.setattr(
        renderer,
        "render_scene",
        lambda index: np.array([[-2.0, 0.0, 2.0]], dtype=np.float32),
    )

    output_dir = tmp_path / "output"
    renderer.generate_random_samples(3, roll_prob=0, savedir=output_dir, n_jobs=1)

    images = sorted((output_dir / "synth_imgs").glob("*.png"))
    masks = sorted((output_dir / "masks").glob("*.png"))
    assert [path.name for path in images] == ["0.png", "1.png", "2.png"]
    assert [path.name for path in masks] == ["0.png", "1.png", "2.png"]
    assert np.asarray(Image.open(images[0])).dtype == np.uint16


def test_draw_scene_returns_cropped_3d_volume_and_retains_thin_mask(
    monkeypatch,
):
    cell = np.zeros((3, 3), dtype=float)
    cell[1, 1] = 1
    volume = np.stack([cell, cell * 2])
    monkeypatch.setattr(colony_simulation, "raster_cell", lambda *args, **kwargs: cell)
    monkeypatch.setattr(
        colony_simulation,
        "rotate",
        lambda image, *args, **kwargs: image,
    )
    monkeypatch.setattr(colony_simulation, "convert_to_3D", lambda image: volume)
    monkeypatch.setattr(
        colony_simulation,
        "get_crop_bounds_2D",
        lambda mask: ((3, 6), (1, 4)),
    )
    simulation = ColonySimulation.__new__(ColonySimulation)
    simulation.scene_shape = (7, 11)

    rendered, mask = simulation.draw_scene(
        [[3, 3, 0, [-4, 0]]],
        as_3D=True,
        crop=True,
    )

    assert rendered.shape == (2, 3, 3)
    assert mask.shape == (3, 3)
    assert rendered[:, 1, 1].tolist() == [1, 2]
    assert mask[1, 1] == 1


def test_colony_3d_rendering_normalizes_the_whole_volume(monkeypatch):
    renderer = ColonyRenderer.__new__(ColonyRenderer)
    renderer.PSF = SimpleNamespace(
        mode="3d fluo",
        kernel=np.stack([np.ones((2, 2)), np.ones((2, 2)) * 3]),
    )
    renderer.resize_amount = 1
    renderer.force_2D = False
    renderer.OPL_loader = lambda index: np.ones((2, 2, 2))
    monkeypatch.setattr(
        colony_renderer,
        "convolve_rescale",
        lambda image, kernel, *args, **kwargs: image * kernel.sum(),
    )

    rendered = renderer.render_scene(0)

    assert rendered.shape == (2, 2, 2)
    assert rendered[0].mean() == pytest.approx(0)
    assert rendered[1].mean() == pytest.approx(1)


def test_2d_convolution_rejects_3d_kernels():
    with pytest.raises(ValueError, match="2-D image and kernel"):
        convolve_rescale(
            np.ones((3, 3)),
            np.ones((2, 3, 3)),
            rescale_factor=1,
            rescale_int=False,
        )
