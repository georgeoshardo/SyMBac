"""Tests for the coherent 4f brightfield reference relay.

Everything here needs JAX and Chromatix, and the project only installs them
for Python 3.12, so the whole file skips if they are missing. It runs in the
3.12 Pixi environment.

Fixed benchmark values, as agreed for comparability:

    wavelength_vacuum_um    = 0.532
    input_dx_um             = 0.3
    focal_length_um         = 1800.0
    medium_refractive_index = 1.33
    objective_na            = 0.3

Small arrays are used so the tests stay fast: (32, 32) square and (24, 32)
non-square. With these values one Fourier-plane pixel is
``lambda * f / (n * N * dx)`` = 2400 / N micrometres, and the objective pupil
radius is ``f * NA / n`` = 406.0 um, so the pupil is resolved by several
pixels and sits well inside the transformed array. Every spatial frequency
used below is an exact multiple of the grid step ``1 / (N * dx)`` and lies
below the input Nyquist frequency ``1 / (2 * dx)`` = 1.667 cycles/um.
"""

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

from chromatix import VectorField  # noqa: E402  (imported after the skip guard)

from SyMBac.brightfield import (  # noqa: E402
    apply_thin_phase_sample,
    scene_to_thickness_um,
)
from SyMBac.drawing import draw_scene_from_segments  # noqa: E402
from tests.reference_models.brightfield_coherent_4f import (  # noqa: E402
    coherent_4f_reference,
)

WAVELENGTH_VACUUM_UM = 0.532
INPUT_DX_UM = 0.3
FOCAL_LENGTH_UM = 1800.0
MEDIUM_REFRACTIVE_INDEX = 1.33
OBJECTIVE_NA = 0.3

# lambda * f / n, the length squared that sets both the Fourier-plane sampling
# and the transform normalization.
LENS_SCALE_UM2 = WAVELENGTH_VACUUM_UM * FOCAL_LENGTH_UM / MEDIUM_REFRACTIVE_INDEX
# Coherent amplitude cutoff of the objective, in cycles per micrometre.
COHERENT_CUTOFF_CYCLES_PER_UM = OBJECTIVE_NA / WAVELENGTH_VACUUM_UM
INPUT_NYQUIST_CYCLES_PER_UM = 1.0 / (2.0 * INPUT_DX_UM)

RELAY_KWARGS = dict(
    focal_length_um=FOCAL_LENGTH_UM,
    medium_refractive_index=MEDIUM_REFRACTIVE_INDEX,
    objective_na=OBJECTIVE_NA,
)


def _vector_field(shape, modulation=None, polarisation_angle=np.pi / 4):
    """Small vector plane wave, optionally multiplied by a 2D pattern."""
    field = cf.plane_wave(
        shape,
        INPUT_DX_UM,
        WAVELENGTH_VACUUM_UM,
        amplitude=cf.linear(polarisation_angle),
        scalar=False,
    )
    if modulation is None:
        return field
    return field.replace(u=jnp.asarray(modulation)[..., None] * field.u)


def _centred_fft2(array):
    """Centred forward transform over the two spatial axes."""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(array, axes=(0, 1)), axes=(0, 1)), axes=(0, 1)
    )


def _independent_relay(u_sample, dy_um, dx_um, *, objective_na=OBJECTIVE_NA):
    """My own centred-FFT model of the relay, so the test does not lean on
    Chromatix to check Chromatix.

    plane 1 -> 2:  u2 = -1j * (dy1 * dx1) / (lambda f / n) * F{u1}
    pupil:         mask = (y2**2 + x2**2) <= (pupil_diameter_um / 2) ** 2
                   pupil_diameter_um = 2 * f * NA / n
    plane 2 -> 3:  u3 = -1j * (dy2 * dx2) / (lambda f / n) * F{u2 * mask}
    """
    height, width = u_sample.shape[:2]

    u_pupil = (-1j * (dy_um * dx_um) / LENS_SCALE_UM2) * _centred_fft2(u_sample)
    dy_pupil = LENS_SCALE_UM2 / (height * dy_um)
    dx_pupil = LENS_SCALE_UM2 / (width * dx_um)

    pupil_diameter_um = (
        2 * FOCAL_LENGTH_UM * objective_na / MEDIUM_REFRACTIVE_INDEX
    )
    y_um = dy_pupil * (np.arange(height) - height / 2)
    x_um = dx_pupil * (np.arange(width) - width / 2)
    inside = (y_um[:, None] ** 2 + x_um[None, :] ** 2) <= (pupil_diameter_um / 2) ** 2

    u_camera = (-1j * (dy_pupil * dx_pupil) / LENS_SCALE_UM2) * _centred_fft2(
        u_pupil * inside[..., None]
    )
    dy_camera = LENS_SCALE_UM2 / (height * dy_pupil)
    dx_camera = LENS_SCALE_UM2 / (width * dx_pupil)
    return {
        "u_camera": u_camera,
        "pupil_dx_um": (dy_pupil, dx_pupil),
        "camera_dx_um": (dy_camera, dx_camera),
        "pupil_diameter_um": pupil_diameter_um,
        "pupil_mask": inside,
    }


def _inverted(array):
    """Turn the image over: out[i, j] = in[-i % H, -j % W]."""
    height, width = array.shape[:2]
    return array[(-np.arange(height)) % height][:, (-np.arange(width)) % width]


# Required test 1: compare the complex camera field with my own FFT model.
def test_camera_field_matches_independent_fourier_benchmark():
    shape = (24, 32)
    rng = np.random.default_rng(20260905)
    modulation = np.exp(1j * rng.uniform(-0.6, 0.6, size=shape)) * (
        1.0 + 0.3 * rng.uniform(-1.0, 1.0, size=shape)
    )
    field = _vector_field(shape, modulation)

    camera = coherent_4f_reference(field, **RELAY_KWARGS)

    dy_um, dx_um = np.asarray(field.dx, dtype=float).reshape(-1)
    expected = _independent_relay(np.asarray(field.u), dy_um, dx_um)["u_camera"]

    got = np.asarray(camera.u)
    scale = float(np.max(np.abs(expected)))
    # Chromatix works in complex64 and my model in float64, so they cannot
    # agree to the last bit. I measured the gap at about 2e-7 of the peak
    # amplitude and set the tolerance well above that.
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5 * scale)


# Required test 2: find the image inversion and write the index rule down
# instead of quietly flipping the output.
def test_image_is_inverted_with_the_stated_index_convention():
    shape = (24, 32)
    height, width = shape
    rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # Asymmetric target, built only from frequencies that fit inside the pupil
    # so nothing is filtered out and the relay is just an inversion.
    target = np.ones(shape)
    for row_order, column_order, phase in ((1, 2, 0.3), (2, 1, 1.1), (0, 3, 2.0)):
        assert (
            np.hypot(
                row_order / (height * INPUT_DX_UM), column_order / (width * INPUT_DX_UM)
            )
            < COHERENT_CUTOFF_CYCLES_PER_UM
        )
        target = target + 0.3 * np.cos(
            2 * np.pi * (row_order * rows / height + column_order * columns / width)
            + phase
        )
    field = _vector_field(shape, target)

    camera = coherent_4f_reference(field, **RELAY_KWARGS)

    sample_u = np.asarray(field.u)
    camera_u = np.asarray(camera.u)

    brightest = np.unravel_index(
        np.argmax(np.abs(sample_u[..., 2])), sample_u.shape[:2]
    )
    expected_index = ((-brightest[0]) % height, (-brightest[1]) % width)
    assert (
        np.unravel_index(np.argmax(np.abs(camera_u[..., 2])), camera_u.shape[:2])
        == expected_index
    )

    # Check the whole field, not only the peak pixel.
    inverted_sample = _inverted(sample_u)
    constant = camera_u[expected_index][2] / inverted_sample[expected_index][2]
    np.testing.assert_allclose(
        camera_u,
        constant * inverted_sample,
        rtol=1e-4,
        atol=1e-4 * float(np.max(np.abs(camera_u))),
    )


# Required test 3: shapes and sampling, square and non-square.
@pytest.mark.parametrize("shape", [(32, 32), (24, 32)])
def test_sampling_returns_to_the_input_sampling(shape):
    field = _vector_field(shape)

    camera = coherent_4f_reference(field, **RELAY_KWARGS)

    assert camera.u.shape == field.u.shape == (shape[0], shape[1], 3)
    assert tuple(camera.spatial_shape) == shape

    dy_um, dx_um = np.asarray(field.dx, dtype=float).reshape(-1)
    reference = _independent_relay(np.asarray(field.u), dy_um, dx_um)

    # equal focal lengths mean magnification one, so the camera sampling comes
    # back to the input sampling
    np.testing.assert_allclose(
        np.asarray(camera.dx, dtype=float).reshape(-1),
        np.asarray(reference["camera_dx_um"]),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(camera.dx, dtype=float).reshape(-1), [dy_um, dx_um], rtol=1e-5
    )

    # and the plane in between is sampled at lambda * f / (n * N * dx)
    pupil_plane = cf.ff_lens(field, FOCAL_LENGTH_UM, MEDIUM_REFRACTIVE_INDEX)
    np.testing.assert_allclose(
        np.asarray(pupil_plane.dx, dtype=float).reshape(-1),
        [
            LENS_SCALE_UM2 / (shape[0] * dy_um),
            LENS_SCALE_UM2 / (shape[1] * dx_um),
        ],
        rtol=1e-5,
    )


# Required test 4: the pupil should pass a frequency below the cutoff and
# stop one above it.
def test_pupil_passes_below_cutoff_and_suppresses_above_cutoff():
    shape = (32, 32)
    height, width = shape
    frequency_step_cycles_per_um = 1.0 / (width * INPUT_DX_UM)

    passed_order, blocked_order = 2, 8
    passed_frequency = passed_order * frequency_step_cycles_per_um  # 0.2083 cyc/um
    blocked_frequency = blocked_order * frequency_step_cycles_per_um  # 0.8333 cyc/um

    # both frequencies land on the grid and stay below the input Nyquist
    assert passed_frequency < COHERENT_CUTOFF_CYCLES_PER_UM < blocked_frequency
    assert blocked_frequency < INPUT_NYQUIST_CYCLES_PER_UM
    assert COHERENT_CUTOFF_CYCLES_PER_UM == pytest.approx(
        OBJECTIVE_NA / WAVELENGTH_VACUUM_UM
    )

    columns = np.arange(width)[None, :] * np.ones((height, 1))
    x_um = columns * INPUT_DX_UM
    target = (
        1.0
        + 0.4 * np.cos(2 * np.pi * passed_frequency * x_um)
        + 0.4 * np.cos(2 * np.pi * blocked_frequency * x_um)
    )
    field = _vector_field(shape, target)

    camera = coherent_4f_reference(field, **RELAY_KWARGS)

    def amplitude_at_order(component, order):
        spectrum = _centred_fft2(component[..., None])[..., 0]
        return abs(spectrum[height // 2, width // 2 + order])

    sample_component = np.asarray(field.u)[..., 2]
    camera_component = np.asarray(camera.u)[..., 2]

    sample_passed = amplitude_at_order(sample_component, passed_order)
    sample_blocked = amplitude_at_order(sample_component, blocked_order)
    camera_passed = amplitude_at_order(camera_component, passed_order)
    camera_blocked = amplitude_at_order(camera_component, blocked_order)

    assert sample_passed > 0.0 and sample_blocked > 0.0
    # the component the objective supports comes through
    assert camera_passed / sample_passed == pytest.approx(1.0, rel=1e-3)
    # the one it does not support is gone
    assert camera_blocked <= 1e-6 * camera_passed


# Required test 5: a uniform scene stays uniform at the camera.
@pytest.mark.parametrize("thickness_um_value", [0.0, 0.4])
def test_uniform_scene_gives_uniform_camera_intensity(thickness_um_value):
    shape = (32, 32)
    field = _vector_field(shape)
    sampled = apply_thin_phase_sample(
        field, np.full(shape, thickness_um_value), refractive_index_difference=0.05
    )

    camera = coherent_4f_reference(sampled, **RELAY_KWARGS)

    intensity = np.asarray(camera.intensity)
    spread = float(intensity.max() - intensity.min()) / float(intensity.mean())
    assert spread < 1e-6


# Required test 6: a real SyMBac scene all the way through to the camera.
def test_symbac_scene_composes_through_sample_and_relay():
    cells_segment_data = [
        {
            "positions": np.array([[12.0, 10.0], [18.0, 10.0]]),
            "radii": np.array([4.0, 4.0]),
            "mask_label": 1,
            "cell_id": 1,
        }
    ]
    scene, _mask = draw_scene_from_segments(cells_segment_data, (24, 32), 0, True)
    assert np.any(scene > 0.0)

    thickness_um = scene_to_thickness_um(
        scene, pix_mic_conv_um=0.065, resize_amount=3
    )
    sampled = apply_thin_phase_sample(
        _vector_field(scene.shape), thickness_um, refractive_index_difference=0.05
    )

    camera = coherent_4f_reference(sampled, **RELAY_KWARGS)

    intensity = np.asarray(camera.intensity)
    assert isinstance(camera, VectorField)
    assert camera.u.shape == (24, 32, 3)
    assert intensity.shape == (24, 32)
    assert np.all(np.isfinite(intensity))
    assert np.all(intensity >= 0.0)


# Required test 7: power. I use the sample plane as the reference, read with
# Field.power.
def test_power_is_conserved_by_a_full_pupil_and_never_increased_by_a_finite_one():
    shape = (32, 32)
    rng = np.random.default_rng(7)
    field = _vector_field(shape, 1.0 + 0.3 * rng.standard_normal(shape))
    sample_power = float(np.asarray(field.power).squeeze())

    dy_um, dx_um = np.asarray(field.dx, dtype=float).reshape(-1)
    pupil_dy_um, pupil_dx_um = _independent_relay(
        np.asarray(field.u), dy_um, dx_um
    )["pupil_dx_um"]
    # the farthest pupil-plane sample the grid holds, i.e. a corner
    corner_radius_um = np.hypot(
        (shape[0] / 2) * pupil_dy_um, (shape[1] / 2) * pupil_dx_um
    )

    full_na = MEDIUM_REFRACTIVE_INDEX
    full_pupil_radius_um = FOCAL_LENGTH_UM * full_na / MEDIUM_REFRACTIVE_INDEX
    assert full_pupil_radius_um > corner_radius_um

    full = coherent_4f_reference(
        field,
        focal_length_um=FOCAL_LENGTH_UM,
        medium_refractive_index=MEDIUM_REFRACTIVE_INDEX,
        objective_na=full_na,
    )
    assert float(np.asarray(full.power).squeeze()) == pytest.approx(
        sample_power, rel=1e-4
    )

    stopped = coherent_4f_reference(field, **RELAY_KWARGS)
    stopped_power = float(np.asarray(stopped.power).squeeze())
    assert stopped_power <= sample_power * (1.0 + 1e-6)
    assert stopped_power > 0.0


# Required test 8: bad input must fail clearly, and the input field must not
# be touched.
@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(focal_length_um=0.0), "focal_length_um"),
        (dict(focal_length_um=-1800.0), "focal_length_um"),
        (dict(focal_length_um=np.nan), "focal_length_um"),
        (dict(medium_refractive_index=0.0), "medium_refractive_index"),
        (dict(medium_refractive_index=-1.33), "medium_refractive_index"),
        (dict(medium_refractive_index=np.inf), "medium_refractive_index"),
        (dict(objective_na=0.0), "objective_na"),
        (dict(objective_na=-0.3), "objective_na"),
        (dict(objective_na=np.nan), "objective_na"),
    ],
)
def test_invalid_optical_parameters_raise(kwargs, message):
    field = _vector_field((8, 8))
    arguments = {**RELAY_KWARGS, **kwargs}
    with pytest.raises(ValueError, match=message):
        coherent_4f_reference(field, **arguments)


def test_na_above_the_medium_index_raises():
    field = _vector_field((8, 8))
    arguments = {**RELAY_KWARGS, "objective_na": MEDIUM_REFRACTIVE_INDEX + 0.01}
    with pytest.raises(ValueError, match="objective_na must satisfy"):
        coherent_4f_reference(field, **arguments)


def test_scalar_field_is_rejected():
    scalar_field = cf.plane_wave((8, 8), INPUT_DX_UM, WAVELENGTH_VACUUM_UM)
    with pytest.raises(TypeError, match="VectorField"):
        coherent_4f_reference(scalar_field, **RELAY_KWARGS)


def test_multi_wavelength_field_is_rejected():
    chromatic_field = cf.plane_wave(
        (8, 8),
        INPUT_DX_UM,
        (jnp.array([0.45, 0.532]), jnp.array([0.5, 0.5])),
        amplitude=cf.linear(0.0),
        scalar=False,
    )
    with pytest.raises((TypeError, ValueError), match="monochromatic"):
        coherent_4f_reference(chromatic_field, **RELAY_KWARGS)


def test_input_field_is_not_modified():
    shape = (16, 16)
    rng = np.random.default_rng(3)
    field = _vector_field(shape, 1.0 + 0.2 * rng.standard_normal(shape))
    before = np.asarray(field.u).copy()
    dx_before = np.asarray(field.dx).copy()

    coherent_4f_reference(field, **RELAY_KWARGS)

    np.testing.assert_array_equal(np.asarray(field.u), before)
    np.testing.assert_array_equal(np.asarray(field.dx), dx_before)
