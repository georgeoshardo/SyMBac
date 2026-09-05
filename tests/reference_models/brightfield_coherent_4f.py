"""Coherent 4f relay, used as a reference for the brightfield work.

This is the relay from the pinned Chromatix 0.6.0 Fourier-ptychography
example: start from the field just after the sample, do one focal transform to
the pupil plane, apply the finite objective pupil there, then do a second
focal transform to the camera plane.

Both lenses get the same focal length and the same surrounding index, so the
magnification is one and the camera sampling comes back to the sample
sampling.

What this is not: it is monochromatic, on-axis, fully coherent and paraxial.
It calls ``cf.ff_lens``, not ``cf.high_na_ff_lens``, so it says nothing about
high-NA vector imaging. There is no defocus, no partial coherence and no
detector model. It fixes the relay convention, it is not the production
microscope.
"""

from __future__ import annotations

import numpy as np

import chromatix.functional as cf
from chromatix import VectorField

__all__ = ["coherent_4f_reference"]


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


def coherent_4f_reference(
    field,
    *,
    focal_length_um,
    medium_refractive_index,
    objective_na,
):
    """Relay a sample-plane vector field to the camera plane of a 4f system.

    The two calls are

    1. ``cf.ff_lens(field, f, n)`` — sample plane to pupil plane;
    2. ``cf.ff_lens(pupil_plane, f, n, NA=objective_na)`` — pupil plane to
       camera plane, with the objective pupil applied on the way in.

    I read the 0.6.0 source to get this right: ``ff_lens`` applies
    ``circular_pupil`` with diameter ``2 * f * NA / n`` to the field it
    receives, and only then transforms it. So giving ``NA`` to the second call
    puts the pupil on the physical Fourier grid made by the first transform,
    which is where the objective aperture belongs.

    Two forward focal transforms turn the image over. The camera sample at
    array index ``(i, j)`` is the sample-plane index
    ``((-i) % height, (-j) % width)``. Nothing here flips it back; the tests
    measure the inversion and registration is left for later.

    Parameters
    ----------
    field : chromatix.VectorField
        Monochromatic sample-plane field, components ordered ``[z, y, x]``.
    focal_length_um : float
        Focal length of both lenses, in micrometres. Positive and finite.
    medium_refractive_index : float
        Index of the surrounding medium, dimensionless. Positive and finite.
    objective_na : float
        Objective numerical aperture, dimensionless, with
        ``0 < objective_na <= medium_refractive_index``.

    Returns
    -------
    chromatix.VectorField
        The camera-plane field, same shape and same ``[z, y, x]`` component
        order as the input. The input field is left alone.
    """
    if not isinstance(field, VectorField):
        raise TypeError(
            "field must be a monochromatic Chromatix VectorField, got "
            f"{type(field).__name__}."
        )

    wavelengths_um = np.asarray(field.spectrum.wavelength, dtype=float).reshape(-1)
    if wavelengths_um.size != 1:
        raise ValueError(
            f"field must be monochromatic, got {wavelengths_um.size} wavelengths."
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

    pupil_plane = cf.ff_lens(field, focal_length_um, medium_refractive_index)
    return cf.ff_lens(
        pupil_plane, focal_length_um, medium_refractive_index, NA=objective_na
    )
