"""Tests for the coherent brightfield microcolony preview.

Everything here needs JAX and Chromatix, and the project only installs them
for Python 3.12, so the whole file skips if they are missing. It runs in the
3.12 Pixi environment.

I keep the arrays small and draw the scenes straight with
``draw_scene_from_segments``, so no growth simulation runs here.

Optical parameters used below, matching the committed example:

    pix_mic_conv_um             = 0.065
    resize_amount               = 3
    wavelength_vacuum_um        = 0.532
    refractive_index_difference = 0.05
    focal_length_um             = 1800.0
    medium_refractive_index     = 1.33
    objective_na                = 0.3

so the sample spacing is 0.065 / 3 = 0.0216667 um.
"""

import importlib
import io
import os
from contextlib import redirect_stdout

import numpy as np
import pytest

jnp = pytest.importorskip(
    "jax.numpy",
    reason="JAX is installed only for Python 3.12 in this project.",
)
cf = pytest.importorskip(
    "chromatix.functional",
    reason="Chromatix is installed only for Python 3.12 in this project.",
)

from SyMBac.brightfield import (  # noqa: E402
    apply_thin_phase_sample,
    scene_to_thickness_um,
)
from SyMBac.drawing import draw_scene_from_segments  # noqa: E402
from examples.brightfield_microcolony_preview import (  # noqa: E402
    invert_to_camera_orientation,
    render_coherent_microcolony_preview,
)
from tests.reference_models.brightfield_coherent_4f import (  # noqa: E402
    coherent_4f_reference,
)

PIX_MIC_CONV_UM = 0.065
RESIZE_AMOUNT = 3
WAVELENGTH_VACUUM_UM = 0.532
REFRACTIVE_INDEX_DIFFERENCE = 0.05
FOCAL_LENGTH_UM = 1800.0
MEDIUM_REFRACTIVE_INDEX = 1.33
OBJECTIVE_NA = 0.3

PREVIEW_KWARGS = dict(
    pix_mic_conv_um=PIX_MIC_CONV_UM,
    resize_amount=RESIZE_AMOUNT,
    wavelength_vacuum_um=WAVELENGTH_VACUUM_UM,
    refractive_index_difference=REFRACTIVE_INDEX_DIFFERENCE,
    focal_length_um=FOCAL_LENGTH_UM,
    medium_refractive_index=MEDIUM_REFRACTIVE_INDEX,
    objective_na=OBJECTIVE_NA,
)

REQUIRED_KEYS = (
    "thickness_um",
    "camera_intensity",
    "blank_intensity",
    "fractional_contrast",
    "camera_mask",
    "camera_dx_um",
)


def _small_scene(shape=(48, 56)):
    """A small two-cell colony from the current drawing pipeline."""
    cells = [
        {
            "positions": np.array([[20.0, 16.0], [30.0, 16.0]]),
            "radii": np.array([6.0, 6.0]),
            "mask_label": 1,
            "cell_id": 1,
        },
        {
            "positions": np.array([[22.0, 32.0], [32.0, 34.0]]),
            "radii": np.array([5.0, 5.0]),
            "mask_label": 2,
            "cell_id": 2,
        },
    ]
    return draw_scene_from_segments(cells, shape, 0, True)


# Task point 1: a drawn scene goes through the whole path and comes back with
# every array at the shape I expect.
def test_drawn_scene_passes_through_the_whole_preview_path():
    scene, mask = _small_scene()

    preview = render_coherent_microcolony_preview(scene, mask, **PREVIEW_KWARGS)

    for key in REQUIRED_KEYS:
        assert key in preview, key
    for key in ("thickness_um", "camera_intensity", "blank_intensity",
                "fractional_contrast", "camera_mask"):
        assert preview[key].shape == scene.shape, key
    assert preview["field_shape"] == (scene.shape[0], scene.shape[1], 3)

    # unit magnification: the camera sampling is the input sampling again
    expected_dx_um = PIX_MIC_CONV_UM / RESIZE_AMOUNT
    np.testing.assert_allclose(preview["sample_dx_um"], expected_dx_um, rtol=1e-6)
    np.testing.assert_allclose(preview["camera_dx_um"], expected_dx_um, rtol=1e-5)


# Task point 2: both intensities are finite and non-negative, and what comes
# back is the calculated data.
def test_raw_and_blank_intensities_are_finite_and_non_negative():
    scene, mask = _small_scene()

    preview = render_coherent_microcolony_preview(scene, mask, **PREVIEW_KWARGS)

    for key in ("camera_intensity", "blank_intensity"):
        values = preview[key]
        assert np.all(np.isfinite(values)), key
        assert np.all(values >= 0.0), key

    # a second identical call gives the same arrays, and nothing has been
    # scaled to a unit maximum
    again = render_coherent_microcolony_preview(scene, mask, **PREVIEW_KWARGS)
    np.testing.assert_array_equal(preview["camera_intensity"], again["camera_intensity"])
    np.testing.assert_array_equal(preview["blank_intensity"], again["blank_intensity"])
    assert preview["camera_intensity"].max() != pytest.approx(1.0, abs=1e-9)


# Task point 3: no sample means no contrast.
def test_zero_thickness_scene_gives_zero_fractional_contrast():
    _, mask = _small_scene()
    zero_scene = np.zeros(mask.shape)

    preview = render_coherent_microcolony_preview(zero_scene, mask, **PREVIEW_KWARGS)

    np.testing.assert_allclose(preview["thickness_um"], 0.0, atol=0.0)
    # with no sample the two runs are identical, so the contrast is zero down
    # to the float32 level Chromatix works in
    assert np.max(np.abs(preview["fractional_contrast"])) <= 1e-6


# Task point 4: the contrast is exactly its equation, and nothing has been
# normalized per image.
def test_fractional_contrast_equals_its_equation_without_normalisation():
    scene, mask = _small_scene()

    preview = render_coherent_microcolony_preview(scene, mask, **PREVIEW_KWARGS)

    camera_intensity = preview["camera_intensity"]
    blank_intensity = preview["blank_intensity"]
    expected = (camera_intensity - blank_intensity) / blank_intensity
    np.testing.assert_array_equal(preview["fractional_contrast"], expected)

    # if anything had been divided by its own maximum these would be 1.0
    assert camera_intensity.max() != pytest.approx(1.0, abs=1e-9)
    assert np.max(np.abs(preview["fractional_contrast"])) != pytest.approx(1.0, abs=1e-9)

    source = open(
        os.path.join("examples", "brightfield_microcolony_preview.py"), encoding="utf8"
    ).read()
    body = source.split("def main(")[0]
    for forbidden in ("np.clip", ".max()", "np.ptp", "histogram"):
        assert forbidden not in body, forbidden


# Task point 5: the camera field matches the reference accepted in PR #71,
# starting from the same post-sample field.
def test_camera_intensity_agrees_with_the_accepted_reference():
    scene, mask = _small_scene()

    preview = render_coherent_microcolony_preview(scene, mask, **PREVIEW_KWARGS)

    thickness_um = scene_to_thickness_um(
        scene, pix_mic_conv_um=PIX_MIC_CONV_UM, resize_amount=RESIZE_AMOUNT
    )
    source_field = cf.plane_wave(
        scene.shape,
        PIX_MIC_CONV_UM / RESIZE_AMOUNT,
        WAVELENGTH_VACUUM_UM,
        power=1.0,
        amplitude=cf.linear(0.0),
        scalar=False,
    )
    post_sample_field = apply_thin_phase_sample(
        source_field,
        thickness_um,
        refractive_index_difference=REFRACTIVE_INDEX_DIFFERENCE,
    )
    reference_camera = coherent_4f_reference(
        post_sample_field,
        focal_length_um=FOCAL_LENGTH_UM,
        medium_refractive_index=MEDIUM_REFRACTIVE_INDEX,
        objective_na=OBJECTIVE_NA,
    )

    reference_intensity = np.asarray(reference_camera.intensity, dtype=float)
    peak = float(np.max(reference_intensity))
    np.testing.assert_allclose(
        preview["camera_intensity"], reference_intensity, rtol=1e-6, atol=1e-6 * peak
    )


# Task point 6: the mask follows the inversion rule exactly and the caller's
# mask is left alone.
def test_camera_mask_follows_the_modular_index_rule_and_input_is_unchanged():
    # single-pixel labels, so there is no doubt where each one lands
    mask = np.zeros((6, 8), dtype=int)
    mask[1, 2] = 7
    mask[4, 5] = 9
    mask_before = mask.copy()
    scene = np.zeros(mask.shape)

    preview = render_coherent_microcolony_preview(scene, mask, **PREVIEW_KWARGS)
    camera_mask = preview["camera_mask"]

    height, width = mask.shape
    expected = np.empty_like(mask)
    for row in range(height):
        for column in range(width):
            expected[row, column] = mask[(-row) % height, (-column) % width]
    np.testing.assert_array_equal(camera_mask, expected)

    for label in (7, 9):
        source_index = tuple(np.argwhere(mask == label)[0])
        assert tuple(np.argwhere(camera_mask == label)[0]) == (
            (-source_index[0]) % height,
            (-source_index[1]) % width,
        )

    np.testing.assert_array_equal(mask, mask_before)

    # For a blob I have to compare pixel sets. Comparing first pixels would be
    # wrong: the first pixel in raster order is not the image of the first
    # pixel of the original. That caught me out while checking this.
    scene_blob, mask_blob = _small_scene()
    blob_preview = render_coherent_microcolony_preview(
        scene_blob, mask_blob, **PREVIEW_KWARGS
    )
    blob_height, blob_width = mask_blob.shape
    mapped = {
        ((-row) % blob_height, (-column) % blob_width)
        for row, column in np.argwhere(mask_blob == 2)
    }
    assert mapped == {
        tuple(pixel) for pixel in np.argwhere(blob_preview["camera_mask"] == 2)
    }


# Task point 7: the scene, mask and thickness inputs stay as they were.
def test_scene_and_mask_inputs_are_not_modified():
    scene, mask = _small_scene()
    scene_before = scene.copy()
    mask_before = mask.copy()

    preview = render_coherent_microcolony_preview(scene, mask, **PREVIEW_KWARGS)

    np.testing.assert_array_equal(scene, scene_before)
    np.testing.assert_array_equal(mask, mask_before)

    # the thickness that comes back is a new array, not a view on the scene
    preview["thickness_um"][0, 0] += 1.0
    np.testing.assert_array_equal(scene, scene_before)


# Task point 8: bad input has to fail with a message that says what is wrong.
@pytest.mark.parametrize(
    "scene, message",
    [
        (np.zeros((4, 4, 2)), "two-dimensional"),
        (np.zeros(16), "two-dimensional"),
        (np.full((4, 4), np.nan), "finite"),
        (np.full((4, 4), -1.0), "non-negative"),
    ],
)
def test_invalid_scene_raises(scene, message):
    mask = np.zeros((4, 4), dtype=int)
    with pytest.raises(ValueError, match=message):
        render_coherent_microcolony_preview(scene, mask, **PREVIEW_KWARGS)


def test_non_two_dimensional_mask_raises():
    with pytest.raises(ValueError, match="two-dimensional"):
        render_coherent_microcolony_preview(
            np.zeros((4, 4)), np.zeros((4, 4, 3)), **PREVIEW_KWARGS
        )


def test_scene_and_mask_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        render_coherent_microcolony_preview(
            np.zeros((4, 4)), np.zeros((4, 5)), **PREVIEW_KWARGS
        )


@pytest.mark.parametrize(
    "override, message",
    [
        (dict(pix_mic_conv_um=0.0), "pix_mic_conv_um"),
        (dict(resize_amount=-3), "resize_amount"),
        (dict(wavelength_vacuum_um=np.nan), "wavelength_vacuum_um"),
        (dict(focal_length_um=0.0), "focal_length_um"),
        (dict(medium_refractive_index=-1.33), "medium_refractive_index"),
        (dict(objective_na=0.0), "objective_na"),
        (dict(objective_na=1.34), "objective_na must satisfy"),
        (dict(refractive_index_difference=np.inf), "refractive_index_difference"),
    ],
)
def test_invalid_physical_parameters_raise(override, message):
    scene, mask = _small_scene((16, 16))
    with pytest.raises(ValueError, match=message):
        render_coherent_microcolony_preview(
            scene, mask, **{**PREVIEW_KWARGS, **override}
        )


def test_blank_field_is_checked_before_division():
    """The blank field is checked for strict positivity before the division."""
    source = open(
        os.path.join("examples", "brightfield_microcolony_preview.py"), encoding="utf8"
    ).read()
    assert "blank_intensity > 0.0" in source
    assert "strictly positive" in source
    # and no epsilon is slipped into the denominator
    assert "1e-" not in source.split("fractional_contrast = ")[1].split("\n")[0]


# Task point 9: importing the module must draw nothing, print nothing and
# write nothing.
def test_importing_the_example_module_is_silent():
    import examples.brightfield_microcolony_preview as module

    before = set(os.listdir("."))
    captured = io.StringIO()
    with redirect_stdout(captured):
        importlib.reload(module)
    assert captured.getvalue() == ""
    assert set(os.listdir(".")) == before

    import matplotlib.pyplot as plt

    assert plt.get_fignums() == []

    source = open(
        os.path.join("examples", "brightfield_microcolony_preview.py"), encoding="utf8"
    ).read()
    guarded = source.split('if __name__ == "__main__":')[0]
    assert "plt.show()" not in guarded.split("def main(")[0]
    assert "print(" not in guarded.split("def main(")[0]


def test_invert_to_camera_orientation_rejects_non_two_dimensional_input():
    with pytest.raises(ValueError, match="two-dimensional"):
        invert_to_camera_orientation(np.zeros((2, 2, 2)))


def test_example_module_imports_nothing_from_tests():
    """The example is meant for users, so it must not lean on the test tree."""
    import ast

    source = open(
        os.path.join("examples", "brightfield_microcolony_preview.py"), encoding="utf8"
    ).read()
    imported = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported, "the module should import something"
    for name in imported:
        assert not name.startswith("tests"), name


def test_display_limits_do_not_change_the_returned_arrays():
    """Picking plot limits is a display choice; the data must not move."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene, mask = _small_scene()
    preview = render_coherent_microcolony_preview(scene, mask, **PREVIEW_KWARGS)
    before = {key: preview[key].copy() for key in REQUIRED_KEYS}

    figure, axes = plt.subplots(1, 3)
    limit = float(np.max(np.abs(preview["fractional_contrast"])))
    axes[0].imshow(preview["camera_intensity"], cmap="gray", vmin=0.0, vmax=1.0)
    image = axes[1].imshow(
        preview["fractional_contrast"], cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    figure.colorbar(image, ax=axes[1])
    axes[2].imshow(preview["camera_mask"], cmap="nipy_spectral")
    plt.close(figure)

    for key in REQUIRED_KEYS:
        np.testing.assert_array_equal(preview[key], before[key], err_msg=key)


def test_main_shows_a_figure_without_saving_a_file(monkeypatch, tmp_path):
    """main() may show a figure, but it must not write a file by itself."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import examples.brightfield_microcolony_preview as module

    shown = {"count": 0}
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shown.__setitem__("count", shown["count"] + 1))
    monkeypatch.chdir(tmp_path)

    captured = io.StringIO()
    with redirect_stdout(captured):
        module.main()

    assert shown["count"] == 1
    assert list(tmp_path.iterdir()) == []
    assert "deterministic drawing-pipeline demonstration" in captured.getvalue().lower()
    plt.close("all")
