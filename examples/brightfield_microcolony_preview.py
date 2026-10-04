"""First coherent brightfield look at a SyMBac microcolony.

This is a developmental preview, not the finished brightfield modality. One
wavelength, one on-axis source, one polarization, fully coherent light, the
paraxial 4f relay checked in PR #71, and exact focus. The result stays on the
supersampled grid and is in relative field intensity, not photons or ADU.

A pure-phase colony at exact focus can come out with very little contrast.
That is the physics, so nothing here adds absorption, defocus, clipping or
normalization to make the cells easier to see.

The path is one chain:

1. SyMBac ``OPL_scene``
2. ``scene_to_thickness_um``
3. an on-axis x-polarized Chromatix ``VectorField``
4. ``apply_thin_phase_sample``
5. ``cf.ff_lens`` to the pupil plane
6. ``cf.ff_lens`` with the objective NA to the camera plane
7. ``VectorField.intensity``
"""

from __future__ import annotations

import numpy as np

from SyMBac.brightfield import apply_thin_phase_sample, scene_to_thickness_um

__all__ = ["render_coherent_microcolony_preview", "invert_to_camera_orientation"]


def _import_chromatix():
    """Import JAX and Chromatix, which this project ships only for Python 3.12."""
    try:
        import jax.numpy as jnp

        import chromatix.functional as cf
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "render_coherent_microcolony_preview requires JAX and Chromatix, "
            "which this project installs only for Python 3.12. Run it from the "
            "Python 3.12 Pixi environment."
        ) from exc
    return cf, jnp


def _finite_2d_array(values, name):
    """Return ``values`` as a finite 2D float array, or raise."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be a two-dimensional array, got {array.ndim} "
            f"dimension(s) with shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _positive_finite(value, name):
    """Return ``value`` as a positive finite float, or raise."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a real number, got {type(value).__name__}."
        ) from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    return number


def invert_to_camera_orientation(array):
    """Turn an array over the way the relay turns the image over.

    PR #71 measured the rule for two forward focal transforms:

    ``camera[i, j] = sample[(-i) % H, (-j) % W]``

    I apply that exact rule here rather than guessing a flip that looks right.
    Only the mask goes through this, so it lines up with the calculated camera
    data; the caller's array is not modified.
    """
    source = np.asarray(array)
    if source.ndim != 2:
        raise ValueError(
            f"mask must be a two-dimensional array, got {source.ndim} "
            f"dimension(s) with shape {source.shape}."
        )
    height, width = source.shape
    return source[(-np.arange(height)) % height][:, (-np.arange(width)) % width]


def _relay_to_camera(cf, field, *, focal_length_um, medium_refractive_index, objective_na):
    """The two lens calls from PR #71: sample plane, pupil plane, camera."""
    pupil_plane = cf.ff_lens(field, focal_length_um, medium_refractive_index)
    return cf.ff_lens(
        pupil_plane, focal_length_um, medium_refractive_index, NA=objective_na
    )


def render_coherent_microcolony_preview(
    opl_scene,
    mask,
    *,
    pix_mic_conv_um,
    resize_amount,
    wavelength_vacuum_um,
    refractive_index_difference,
    focal_length_um,
    medium_refractive_index,
    objective_na,
):
    """Render one coherent camera-plane preview of a SyMBac scene.

    The source is an on-axis, x-polarized plane wave of unit power, sampled at
    ``pix_mic_conv_um / resize_amount`` micrometres. The same source and the
    same relay run a second time with a zero-thickness sample to give the
    blank field, and then

    ``fractional_contrast = (camera_intensity - blank_intensity) / blank_intensity``

    I check that the blank field is finite and strictly positive before that
    division instead of adding an epsilon, because an epsilon would quietly
    paper over a blank field that is actually broken.

    Parameters
    ----------
    opl_scene : array_like
        2D SyMBac scene of projected thickness in supersampled pixels. Finite
        and non-negative.
    mask : array_like
        2D instance or binary mask on the same grid as ``opl_scene``. Comes
        back in camera orientation; the input is left alone.
    pix_mic_conv_um : float
        Pixel size of the final image, micrometres per pixel.
    resize_amount : float
        Supersampling factor, dimensionless.
    wavelength_vacuum_um : float
        Vacuum wavelength, micrometres.
    refractive_index_difference : float
        Index contrast of the cells against the medium, dimensionless. May be
        negative, must be finite.
    focal_length_um : float
        Focal length of both lenses, micrometres.
    medium_refractive_index : float
        Index of the surrounding medium, dimensionless.
    objective_na : float
        Objective NA, with ``0 < objective_na <= medium_refractive_index``.

    Returns
    -------
    dict
        ``thickness_um``, ``camera_intensity``, ``blank_intensity``,
        ``fractional_contrast``, ``camera_mask`` and ``camera_dx_um``, plus
        ``sample_dx_um``, ``pupil_dx_um`` and ``field_shape``. Every array is
        the calculated one: nothing is normalized, clipped or shifted.
    """
    cf, jnp = _import_chromatix()

    scene = _finite_2d_array(opl_scene, "opl_scene")
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(
            f"mask must be a two-dimensional array, got {mask_array.ndim} "
            f"dimension(s) with shape {mask_array.shape}."
        )
    if mask_array.shape != scene.shape:
        raise ValueError(
            "mask and opl_scene must have the same shape, got "
            f"{mask_array.shape} and {scene.shape}."
        )

    pix_mic_conv_um = _positive_finite(pix_mic_conv_um, "pix_mic_conv_um")
    resize_amount = _positive_finite(resize_amount, "resize_amount")
    wavelength_vacuum_um = _positive_finite(
        wavelength_vacuum_um, "wavelength_vacuum_um"
    )
    focal_length_um = _positive_finite(focal_length_um, "focal_length_um")
    medium_refractive_index = _positive_finite(
        medium_refractive_index, "medium_refractive_index"
    )
    objective_na = _positive_finite(objective_na, "objective_na")
    if objective_na > medium_refractive_index:
        raise ValueError(
            "objective_na must satisfy 0 < objective_na <= "
            f"medium_refractive_index, got {objective_na} > "
            f"{medium_refractive_index}."
        )
    try:
        delta_n = float(refractive_index_difference)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "refractive_index_difference must be a real number, got "
            f"{type(refractive_index_difference).__name__}."
        ) from exc
    if not np.isfinite(delta_n):
        raise ValueError(
            "refractive_index_difference must be finite, got "
            f"{refractive_index_difference!r}."
        )

    thickness_um = scene_to_thickness_um(
        scene, pix_mic_conv_um=pix_mic_conv_um, resize_amount=resize_amount
    )

    # One source, used by both runs so the blank really is a fair reference.
    dx_um = pix_mic_conv_um / resize_amount
    source_field = cf.plane_wave(
        scene.shape,
        dx_um,
        wavelength_vacuum_um,
        power=1.0,
        amplitude=cf.linear(0.0),  # [E_z, E_y, E_x] = [0, 0, 1]
        scalar=False,
    )

    relay_kwargs = dict(
        focal_length_um=focal_length_um,
        medium_refractive_index=medium_refractive_index,
        objective_na=objective_na,
    )

    sample_field = apply_thin_phase_sample(
        source_field, thickness_um, refractive_index_difference=delta_n
    )
    camera_field = _relay_to_camera(cf, sample_field, **relay_kwargs)

    # Same source, same relay, nothing in the way.
    blank_sample_field = apply_thin_phase_sample(
        source_field, np.zeros_like(thickness_um), refractive_index_difference=delta_n
    )
    blank_field = _relay_to_camera(cf, blank_sample_field, **relay_kwargs)

    camera_intensity = np.asarray(camera_field.intensity, dtype=float)
    blank_intensity = np.asarray(blank_field.intensity, dtype=float)

    if not np.all(np.isfinite(blank_intensity)):
        raise ValueError(
            "the calculated blank intensity contains non-finite values, so "
            "fractional contrast cannot be formed."
        )
    if not np.all(blank_intensity > 0.0):
        raise ValueError(
            "the calculated blank intensity must be strictly positive before "
            f"dividing, but its minimum is {blank_intensity.min()!r}."
        )

    fractional_contrast = (camera_intensity - blank_intensity) / blank_intensity

    pupil_plane = cf.ff_lens(sample_field, focal_length_um, medium_refractive_index)

    return {
        "thickness_um": thickness_um,
        "camera_intensity": camera_intensity,
        "blank_intensity": blank_intensity,
        "fractional_contrast": fractional_contrast,
        "camera_mask": invert_to_camera_orientation(mask_array),
        "camera_dx_um": np.asarray(camera_field.dx, dtype=float).reshape(-1),
        "sample_dx_um": np.asarray(source_field.dx, dtype=float).reshape(-1),
        "pupil_dx_um": np.asarray(pupil_plane.dx, dtype=float).reshape(-1),
        "field_shape": tuple(camera_field.u.shape),
    }


def _deterministic_colony_segments():
    """Four segment-chain cells, written down so the example repeats exactly.

    This is the drawing pipeline, not a growth simulation: I placed the cells
    by hand and left background around them.
    """
    return [
        {
            "positions": np.array([[60.0, 55.0], [78.0, 55.0], [96.0, 55.0]]),
            "radii": np.array([11.0, 11.0, 11.0]),
            "mask_label": 1,
            "cell_id": 1,
        },
        {
            "positions": np.array([[62.0, 92.0], [80.0, 94.0], [98.0, 96.0]]),
            "radii": np.array([10.0, 10.0, 10.0]),
            "mask_label": 2,
            "cell_id": 2,
        },
        {
            "positions": np.array([[118.0, 68.0], [130.0, 80.0]]),
            "radii": np.array([9.0, 9.0]),
            "mask_label": 3,
            "cell_id": 3,
        },
        {
            "positions": np.array([[46.0, 118.0], [64.0, 120.0]]),
            "radii": np.array([9.5, 9.5]),
            "mask_label": 4,
            "cell_id": 4,
        },
    ]


def main():
    """Draw the colony, run the preview, show the four panels."""
    import matplotlib.pyplot as plt

    from SyMBac.drawing import draw_scene_from_segments

    pix_mic_conv_um = 0.065
    resize_amount = 3
    wavelength_vacuum_um = 0.532
    refractive_index_difference = 0.05
    focal_length_um = 1800.0
    medium_refractive_index = 1.33
    objective_na = 0.3

    scene, mask = draw_scene_from_segments(
        _deterministic_colony_segments(), (180, 180), 0, True
    )

    preview = render_coherent_microcolony_preview(
        scene,
        mask,
        pix_mic_conv_um=pix_mic_conv_um,
        resize_amount=resize_amount,
        wavelength_vacuum_um=wavelength_vacuum_um,
        refractive_index_difference=refractive_index_difference,
        focal_length_um=focal_length_um,
        medium_refractive_index=medium_refractive_index,
        objective_na=objective_na,
    )

    camera_intensity = preview["camera_intensity"]
    fractional_contrast = preview["fractional_contrast"]
    print("Deterministic drawing-pipeline demonstration, not a growth simulation.")
    print(f"scene shape: {scene.shape}, sample dx: {preview['sample_dx_um']} um")
    print(f"pupil-plane dx: {preview['pupil_dx_um']} um")
    print(f"camera dx: {preview['camera_dx_um']} um")
    print(
        "raw camera intensity: min {:.6e} max {:.6e} mean {:.6e}".format(
            camera_intensity.min(), camera_intensity.max(), camera_intensity.mean()
        )
    )
    print(
        "fractional contrast: min {:.6e} max {:.6e}".format(
            fractional_contrast.min(), fractional_contrast.max()
        )
    )

    figure, axes = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)
    figure.suptitle(
        "Coherent brightfield preview of a deterministic drawing-pipeline colony "
        "(not a growth simulation): monochromatic, on-axis, one polarization, "
        "paraxial 4f relay, exact focus"
    )

    thickness_image = axes[0].imshow(preview["thickness_um"], cmap="viridis")
    axes[0].set_title("Projected geometric thickness")
    figure.colorbar(thickness_image, ax=axes[0], label="thickness (µm)")

    intensity_image = axes[1].imshow(camera_intensity, cmap="gray")
    axes[1].set_title("Raw camera-plane intensity")
    figure.colorbar(
        intensity_image, ax=axes[1], label="relative field intensity (arb. units)"
    )

    # Symmetric limits, so zero contrast sits in the middle of the colormap.
    contrast_limit = float(np.max(np.abs(fractional_contrast)))
    contrast_image = axes[2].imshow(
        fractional_contrast, cmap="RdBu_r", vmin=-contrast_limit, vmax=contrast_limit
    )
    axes[2].set_title("Fractional contrast")
    figure.colorbar(
        contrast_image,
        ax=axes[2],
        label=f"(I - I_blank) / I_blank, limits ±{contrast_limit:.2e}",
    )

    mask_image = axes[3].imshow(preview["camera_mask"], cmap="nipy_spectral")
    axes[3].set_title("Camera-oriented instance mask")
    figure.colorbar(mask_image, ax=axes[3], label="instance label")

    for axis in axes:
        axis.set_xlabel("supersampled pixel")
        axis.set_ylabel("supersampled pixel")

    plt.show()


if __name__ == "__main__":
    main()
