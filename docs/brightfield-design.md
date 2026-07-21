# Chromatix brightfield design

## Goal

Chromatix will perform the production optical calculation from illumination-field construction to camera-plane intensity.

## Non-goals

This task does not implement:

* Production code.
* A custom CTF/WOTF renderer.
* BPM or multislice propagation.
* Sensor noise.
* Mother-machine device optics.
* Parameter fitting.
* Renderer integration.

## Existing SyMBac input

`OPL_scene` arrays are generated natively within the SyMBac drawing pipeline (e.g., via `draw_simulation_OPL` and `draw_scene_from_segments`). Physically, these values represent proportional cell heights or raw optical path lengths derived from the cell geometries before physical scaling is applied.

To convert these raw array values into physical micrometres, they are multiplied by the spatial resolution `pix_mic_conv` and divided by the upscaling factor `resize_amount`.

## Physical specimen model

```python
thickness_um =
    OPL_scene * pix_mic_conv / resize_amount

transmission =
    exp(-absorption)
    * exp(1j * 2*pi/wavelength_um
          * refractive_index_difference
          * thickness_um)

```

Term Definitions:

* `thickness_um`: Physical thickness of the specimen. Unit: micrometres. Array shape: [H, W]
* `OPL_scene`: Raw optical path length array. Unit: dimensionless. Array shape: [H, W]
* `pix_mic_conv`: Spatial scale of a pixel. Unit: micrometres per pixel. Array shape: scalar
* `resize_amount`: Interpolation scaling factor. Unit: dimensionless. Array shape: scalar
* `transmission`: Complex transmitted optical field. Unit: dimensionless. Array shape: [H, W]
* `absorption`: Absorption coefficient. Unit: dimensionless. Array shape: [H, W]
* `wavelength_um`: Illumination wavelength. Unit: micrometres. Array shape: scalar
* `refractive_index_difference`: Difference in index between cell and medium. Unit: dimensionless. Array shape: scalar

## Chromatix optical path

* plane wave: Proposed function is `chromatix.elements.PlaneWave` (Documentation: [https://chromatix.readthedocs.io](https://chromatix.readthedocs.io)).
* sample amplitude: Proposed function is `chromatix.elements.AmplitudeMask` (Documentation: [https://chromatix.readthedocs.io](https://chromatix.readthedocs.io)).
* sample phase: Proposed function is `chromatix.elements.PhaseMask` (Documentation: [https://chromatix.readthedocs.io](https://chromatix.readthedocs.io)).
* objective pupil and defocus: Proposed function is `chromatix.elements.MicroscopeObjective` (Documentation: [https://chromatix.readthedocs.io](https://chromatix.readthedocs.io)).
* camera-plane field: The resulting complex field propagated to the sensor plane natively by the objective element.
* intensity: Proposed function is `chromatix.elements.IntensitySensor` (Documentation: [https://chromatix.readthedocs.io](https://chromatix.readthedocs.io)).

Chromatix will perform the production optical propagation. It will not be used only for sensor noise.

## Partial-coherence model

One source point is propagated coherently because light originating from a single infinitesimal point on the condenser emits a deterministic plane wave with a constant phase relationship across the entire sample.
Intensities rather than complex fields are summed across mutually incoherent source points because points on an extended source are statistically independent; their rapidly fluctuating cross-interference terms average to zero over the camera's integration time.
Source weights are normalised such that their sum equals exactly 1 to ensure that the total incident energy is conserved regardless of the number of source points sampled.
Source-sampling convergence will be checked by increasing the number of simulated condenser points and verifying that the maximum difference between the resulting intensity images falls below a predefined numerical tolerance threshold.

## Sensor model

The sensor model will output the ideal continuous intensity mapped to a discrete pixel grid. It will not include physical read noise or shot noise, as sensor noise is explicitly defined as a non-goal for this task.

## Physical parameters and units

| Physical quantity | Proposed code name | Value | Unit | Meaning | Source |
| --- | --- | --- | --- | --- | --- |
| Illumination wavelength | `wavelength_um` | TBD | micrometres | Wavelength of light | User input |
| Focal plane offset | `defocus_um` | TBD | micrometres | Defocus distance | User input |
| Numerical aperture | `objective_na` | TBD | N/A | NA of the objective | User input |
| Background index | `sample_medium_refractive_index` | TBD | N/A | Medium refractive index | User input |
| Specimen index | `cell_refractive_index` | TBD | N/A | Cell refractive index | User input |

## Validation tests

* Zero refractive-index contrast: Expected result is a completely uniform, flat image. Failure would mean the model is injecting artifactual phase or amplitude changes.
* A uniform sample: Expected result is flat, constant intensity across the field. Failure would mean propagation steps are introducing non-uniformities or boundary errors.
* One on-axis source: Expected result is a standard coherent imaging response with prominent edge ringing. Failure would mean coherent limits are not accurately modeled.
* Positive and negative defocus: Expected result is a physically accurate, asymmetric intensity change. Failure would mean the defocus phase operator has an incorrect sign or scaling.
* Increasing condenser-source samples: Expected result is asymptotic convergence to a smooth, partially coherent image. Failure would mean incorrect incoherent summation or invalid weight normalisation.
* Padding and cropping: Expected result is that the central region of interest remains invariant regardless of boundary size. Failure would mean edge artifacts or wrap-around errors are polluting the center.
* Constant incident photon scaling: Expected result is total energy conservation across operations. Failure would mean scaling factors are lost during FFTs or discrete propagations.

## Proposed production files

* `SyMBac/brightfield.py`
* `tests/test_brightfield.py`
* `docs/brightfield.md`

## Chromatix dependency strategy

We will pin an exact version or commit hash of chromatix in pixi.toml to guarantee identical environments. The module will target Python 3.12. Initially, Chromatix will be kept as an optional dependency so as not to disrupt existing CI pipelines. The dependency strategy must ensure that the Pixi lockfile resolves successfully across all supported platforms and correctly handles jax and jaxlib dependencies.

## Known assumptions and limitations

* Assumes scalar optics, completely ignoring polarization and vectorial diffraction effects.
* Assumes the thin phase and amplitude object approximation, meaning diffraction within the cell volume itself is ignored.
* Neglects multiple scattering events between densely packed specimens.