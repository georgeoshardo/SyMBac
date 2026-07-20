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
In the existing SyMBac architecture, `OPL_scenes` are generated within simulation loop processes. Physically, these values represent the integrated 2D projection of the specimen's physical thickness in pixel units. 
* `pix_mic_conv` is the scalar conversion factor representing the physical size of a single simulated pixel.
* `resize_amount` is the super-resolution scaling factor applied to the simulation grid.
Together, multiplying the scene by `pix_mic_conv` and dividing by `resize_amount` converts these pixel-space values into absolute physical thickness in micrometres.

## Physical specimen model
```python
thickness_um =
    OPL_scene * pix_mic_conv / resize_amount
transmission =
    exp(-absorption)
    * exp(1j * 2*pi/wavelength_um
          * refractive_index_difference
          * thickness_um)