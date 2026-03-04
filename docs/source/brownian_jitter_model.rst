Brownian Jitter Model
=====================

This page describes the Brownian-style cell jitter model used by
``SyMBac.simulation.Simulation``.

Motivation
----------

In real mother-machine movies, cells are not perfectly static between frames.
Small hydrodynamic and thermal perturbations cause slight translations and
rotations, even when growth/division dynamics are slow.

To reproduce this, SyMBac applies a low-amplitude, temporally correlated
rigid-body perturbation to each cell once per output frame.

Model
-----

For each cell and each frame, SyMBac samples a 3D jitter state:

.. math::

   \mathbf{s}_t = \begin{bmatrix}
   \Delta y_t \\
   \Delta x_t \\
   \Delta \theta_t
   \end{bmatrix}

with an AR(1) / discrete OU-style update:

.. math::

   \mathbf{s}_t = \rho \mathbf{s}_{t-1} +
   \sqrt{1-\rho^2}\,\boldsymbol{\eta}_t

where:

.. math::

   \boldsymbol{\eta}_t \sim \mathcal{N}\!\left(
   \mathbf{0},
   \mathrm{diag}(\sigma_\parallel^2,\sigma_\perp^2,\sigma_\theta^2)
   \right)

and:

* :math:`\rho` is ``brownian_persistence`` in ``[0, 1)``.
* :math:`\sigma_\parallel` is ``brownian_longitudinal_std``.
* :math:`\sigma_\perp` is ``brownian_transverse_std``.
* :math:`\sigma_\theta` is ``brownian_rotation_std``.

Rigid-body update
-----------------

Let :math:`\mathbf{r}_{i,t}` be the position of segment :math:`i`, and
:math:`\mathbf{c}_t` the cell centroid for frame :math:`t`.
The jittered segment position is:

.. math::

   \mathbf{r}'_{i,t} =
   \mathbf{c}_t +
   \mathbf{R}(\Delta\theta_t)\left(\mathbf{r}_{i,t}-\mathbf{c}_t\right) +
   \begin{bmatrix}\Delta x_t \\ \Delta y_t\end{bmatrix}

Each segment body angle is also incremented by :math:`\Delta\theta_t`.

This preserves cell shape while introducing physically plausible motion.

Units and scaling
-----------------

User-facing translation parameters are in microns per frame.
Internally they are converted to simulation pixels via:

.. math::

   \sigma_{\mathrm{px}} = \sigma_{\mu m}\times
   \left(\frac{1}{\mathrm{pix\_mic\_conv}}\right)\times
   \mathrm{resize\_amount}

Rotation is already in radians and does not require pixel scaling.

Parameters in ``Simulation``
----------------------------

* ``brownian_longitudinal_std``: longitudinal translation std (microns/frame).
* ``brownian_transverse_std``: transverse translation std (microns/frame).
* ``brownian_rotation_std``: angular std (radians/frame).
* ``brownian_persistence``: temporal correlation in ``[0, 1)``.
* ``brownian_max_dx_fraction_of_trench_width``: hard cap on ``|dx|`` as a
  fraction of trench width.
* ``brownian_max_dy_fraction_of_segment_radius``: hard cap component on
  ``|dy|`` based on segment radius.
* ``brownian_max_dy_px_floor``: floor (pixels) for the ``|dy|`` cap.
* ``brownian_max_dtheta``: hard cap on ``|dtheta|`` in radians.
* ``brownian_backoff_attempts``: number of geometric backoff retries before rollback.

Safety guards (caps + backoff)
------------------------------

After stochastic sampling, SyMBac applies hard clipping:

.. math::

   |\Delta x_t| \le \alpha_x W_{\mathrm{trench}}

.. math::

   |\Delta y_t| \le \max\!\left(y_{\min}, \alpha_y r_{\mathrm{seg}}\right)

.. math::

   |\Delta \theta_t| \le \theta_{\max}

with:

* :math:`\alpha_x =` ``brownian_max_dx_fraction_of_trench_width``
* :math:`\alpha_y =` ``brownian_max_dy_fraction_of_segment_radius``
* :math:`y_{\min} =` ``brownian_max_dy_px_floor``
* :math:`\theta_{\max} =` ``brownian_max_dtheta``

Then SyMBac validates trench-bound geometry. If invalid, it retries with
backoff scales :math:`1, 1/2, 1/4, ...` up to
``brownian_backoff_attempts``. If all retries fail, the frame jitter is
rolled back for that cell.

Behavioral notes
----------------

* ``brownian_persistence = 0`` gives white-noise-like frame-to-frame jitter.
* Larger persistence (e.g. ``0.8-0.95``) gives smooth drifting motion.
* Setting all three std values to zero disables this jitter.
* This model is independent from per-segment force noise
  (``CellConfig.NOISE_STRENGTH``), and acts as a frame-level rigid-body perturbation.
* Guard parameters prevent large jumps that can tunnel through trench walls.

Practical starting values
-------------------------

For subtle but visible motion in mother-machine data:

* ``brownian_longitudinal_std = 0.01`` to ``0.03``
* ``brownian_transverse_std = 0.005`` to ``0.02``
* ``brownian_rotation_std = 0.002`` to ``0.01``
* ``brownian_persistence = 0.8`` to ``0.9``
* ``brownian_max_dx_fraction_of_trench_width = 0.1`` to ``0.25``
* ``brownian_max_dtheta = 0.01`` to ``0.04``
* ``brownian_backoff_attempts = 3`` to ``6``

Increase gradually to avoid unrealistic slipping or excessive spinning.
