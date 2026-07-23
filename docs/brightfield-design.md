## 1. One Production Model
SyMBac will implement a single, vectorial Chromatix brightfield path for its production model. This is not a scalar approximation. Chromatix will perform the complete physical optical calculation from the illumination-field construction to the camera-plane intensity. It will not be used merely as an additive sensor noise step.

## 2. The Actual SyMBac Specimen Input
Despite its legacy name, `OPL_scene` stores the projected geometric cell thickness in supersampled simulation pixels, not the optical path length. The conversion to physical thickness is defined as:

```python
thickness_um = OPL_scene * pix_mic_conv / resize_amount

```

* `OPL_scene`: 2D array, shape `[H, W]`, dimensionless. Overlapping projected cells are currently treated with a winner-takes-all depth projection; overlap pixels are not additive thickness.
* `pix_mic_conv`: Scalar, physical size of a camera pixel in micrometres (um).
* `resize_amount`: Scalar, dimensionless upscaling factor.
* `thickness_um`: 2D array, shape `[H, W]`, physical thickness (um).

For the initial model, we assume a thin, isotropic, non-birefringent specimen with unit amplitude transmittance and a phase shift defined by:

```python
phase_rad = (
    2 * np.pi / wavelength_vacuum_um
    * refractive_index_difference
    * thickness_um
)
transmission = np.exp(1j * phase_rad)

```

Because the specimen is strictly isotropic and non-birefringent, this single scalar `transmission` value can multiply every electric-field component. The propagation remains vectorial, but the physical specimen interaction delays all polarisation components equally without requiring an imaginary refractive index, birefringence, or 3D BPM parameters.

## 3. The Vector and Polarisation Reasoning

Chromatix represents physical vector fields using the component order `[E_z, E_y, E_x]`. Physical direction coordinates will be defined as spatial frequencies `kykx = [k_y, k_x]` in rad/um.

For each oblique source direction represented by wavevector $\vec{k}$, we construct two orthogonal transverse Jones states ($\vec{E}_1$ and $\vec{E}_2$), verifying transversality with $\vec{k} \cdot \vec{E} = 0$. To represent nominally unpolarised illumination, we propagate these orthogonal states separately through the entire system and average their intensities once at the detector:

```python
intensity_q = 0.5 * intensity_q_1 + 0.5 * intensity_q_2

```

We do not add the two complex fields, nor do we halve both field amplitudes (which would quarter the power). Incident power is explicitly normalised so that changing the arbitrary stable input basis ($\vec{e}_1, \vec{e}_2$), the source angle, or array padding does not change the stated illumination power. The stable input basis is distinguished from the physical interface TE/TM (`s`/`p`) coordinates.

## 4. An Honest Chromatix Optical Path

This design utilizes APIs from the pinned Chromatix 0.6.0 release. The operations will include verified vector sources and intensity averaging. We will investigate `chromatix.elements.high_na_ff_lens`; however, because its documented example is strictly pupil-to-focus, it is not a complete transmitted-light microscope path.

**Unresolved Items (Supervisor Gates):**

* **Object-to-Pupil and Objective Support:** How to handle physical collection and transversality after the spatially varying sample interface. *Proposal:* Benchmark against an established vectorial rigorous coupled-wave analysis (RCWA) reference.
* **Sample/Coverslip/Immersion Interfaces:** Defining transmission Fresnel coefficients. *Proposal:* Validate against specific physical optics derivations (e.g., Novotny & Hecht) before implementation.
* **Physical Defocus:** Applying `chromatix.functional.defocus` correctly to the high-NA vectorial field. *Proposal:* Validate against a low-NA paraxial limit benchmark.

## 5. Parameters and Units

| Code Name | Value | Unit | Physical Meaning | Source / Blocked Claim if unknown |
| --- | --- | --- | --- | --- |
| `wavelength_vacuum_um` | `unknown` | um | Vacuum wavelength | Blocks spectral weights / phase |
| `na_objective` | `unknown` | dimensionless | Numerical aperture | Blocks high-NA collection limits |
| `n_medium` | `unknown` | dimensionless | Refractive index of sample medium | Blocks phase and interface scaling |
| `n_cell` | `unknown` | dimensionless | Refractive index of cell | Blocks phase difference |
| `n_coverslip` | `unknown` | dimensionless | Refractive index of coverslip | Blocks interface calculations |
| `n_immersion` | `unknown` | dimensionless | Refractive index of immersion | Blocks pupil mapping |
| `pix_mic_conv` | `unknown` | um | Camera pixel size | Blocks physical scaling |
| `resize_amount` | `unknown` | dimensionless | Simulation supersampling | Blocks physical scaling |
| `defocus_um` | `unknown` | um | Physical defocus distance | Blocks 3D stack generation |

## 6. Validation Plan

Each claim must be verified against independent references. Benchmarks are marked as pending supervisor approval.

* **Zero contrast and uniform-sample limits:** Flat phase fields must yield uniform intensity (Tolerance: $1 \times 10^{-6}$).
* **Transverse and orthogonal polarisation states:** Must have equal incident power.
* **Invariance to transverse polarisation basis:** Changing the arbitrary input basis must not alter the averaged unpolarised intensity.
* **Vectorial propagation case:** Independent benchmark (pending supervisor approval).
* **Convergence:** Array padding, cropping, and source/spectral sampling must converge asymptoticaly.
* **Low-NA scalar comparison:** Test only; must match scalar FFT propagation strictly within its stated paraxial validity range.
* **Consistent physical scaling:** Must be verified before detector noise is added.

## 7. Staged Implementation Plan

* **Vectorial Core:** Build the smallest test file to construct orthogonal Jones states and normalise power. (Independent vectorial benchmark).
* **Optional Chromatix Dependency:** Wrap Chromatix imports in `try/except ImportError` so it remains optional in the default SyMBac environment.
* **Specimen Integration:** Add the scaled `thickness_um` and thin-phase object operations.
* **Microscope Assembly:** Integrate high-NA collection and interfaces only after supervisor gates on unresolved items are passed.
