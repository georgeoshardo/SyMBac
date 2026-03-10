Brownian Jitter Model
=====================

This page describes the Brownian-style cell jitter model used by
``SyMBac.simulation.Simulation``.

Motivation
----------

In real mother-machine movies, cells are not perfectly static between frames.
Small hydrodynamic and thermal perturbations cause slight translations and
rotations, even when growth/division dynamics are slow.

SyMBac models this with a low-amplitude, temporally correlated rigid-body
perturbation per cell.

Stochastic State Model
----------------------

For each cell and output frame, SyMBac samples:

.. math::

   \mathbf{s}_t =
   \begin{bmatrix}
   \Delta y_t \\
   \Delta x_t \\
   \Delta \theta_t
   \end{bmatrix}

with AR(1) / discrete OU update:

.. math::

   \mathbf{s}_t = \rho\,\mathbf{s}_{t-1} +
   \sqrt{1-\rho^2}\,\boldsymbol{\eta}_t,
   \quad
   \boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0},
   \mathrm{diag}(\sigma_\parallel^2,\sigma_\perp^2,\sigma_\theta^2)).

Where:

* :math:`\rho` is ``brownian_persistence`` in ``[0, 1)``.
* :math:`\sigma_\parallel` is ``brownian_longitudinal_std``.
* :math:`\sigma_\perp` is ``brownian_transverse_std``.
* :math:`\sigma_\theta` is ``brownian_rotation_std``.

Application Modes
-----------------

``brownian_application_mode`` controls how sampled jitter is applied.

``teleport``
~~~~~~~~~~~~

A rigid-body pose update is applied directly:

.. math::

   \mathbf{r}'_{i,t} =
   \mathbf{c}_t +
   \mathbf{R}(\Delta\theta_t)(\mathbf{r}_{i,t}-\mathbf{c}_t) +
   \begin{bmatrix}\Delta x_t \\ \Delta y_t\end{bmatrix}

and segment angle is incremented by :math:`\Delta\theta_t`.

``velocity``
~~~~~~~~~~~~

Jitter is converted into per-body velocity and angular velocity before
sub-stepping, so Pymunk resolves contacts continuously.

Let :math:`T_f = N_{sub}\,\Delta t` be one output-frame duration.

.. math::

   \mathbf{v}_{cm} = \frac{1}{T_f}
   \begin{bmatrix}\Delta x_t \\ \Delta y_t\end{bmatrix},
   \qquad
   \omega = \frac{\Delta\theta_t}{T_f}

For segment :math:`i` with relative offset
:math:`\mathbf{q}_i = \mathbf{r}_i - \mathbf{c}`:

.. math::

   \mathbf{v}_i^* = \mathbf{v}_{cm} +
   \omega
   \begin{bmatrix}
   -q_{i,y} \\
   q_{i,x}
   \end{bmatrix}

SyMBac adds :math:`\mathbf{v}_i^*` to each segment velocity and adds
:math:`\omega` to each segment angular velocity.

``impulse``
~~~~~~~~~~~

Same target field as ``velocity``, but translation is applied as impulse:

.. math::

   \mathbf{J}_i = m_i\left(\mathbf{v}_i^* - \mathbf{v}_i\right)

followed by angular-velocity perturbation.

Units and Scaling
-----------------

User-facing translation parameters are in microns per frame.
Internally:

.. math::

   \sigma_{px} = \sigma_{\mu m}
   \left(\frac{1}{\mathrm{pix\_mic\_conv}}\right)
   \mathrm{resize\_amount}

Rotation is already in radians.

Safety Guards: Caps and Backoff
--------------------------------

SyMBac clips sampled jitter:

.. math::

   |\Delta x_t| \le \alpha_x W_{trench}

.. math::

   |\Delta y_t| \le \max(y_{min}, \alpha_y r_{seg})

.. math::

   |\Delta\theta_t| \le \theta_{max}

with:

* :math:`\alpha_x =` ``brownian_max_dx_fraction_of_trench_width``
* :math:`\alpha_y =` ``brownian_max_dy_fraction_of_segment_radius``
* :math:`y_{min} =` ``brownian_max_dy_px_floor``
* :math:`\theta_{max} =` ``brownian_max_dtheta``

Then SyMBac validates trench-bound geometry. If invalid, it retries with
scales :math:`1, 1/2, 1/4, ...` up to ``brownian_backoff_attempts``.
If all retries fail, that frame's Brownian perturbation is skipped for the
cell.

Projection Safety Net (Dynamic Modes)
-------------------------------------

For ``velocity`` and ``impulse`` modes, a post-step safety pass projects any
out-of-bounds segment center to the nearest valid trench interior point for
the side walls and closed trench end (the open end is intentionally left
unconstrained):

.. math::

   \mathbf{r}_i \leftarrow \Pi_{\Omega}(\mathbf{r}_i)

with :math:`\Omega` the radius-aware valid trench domain.

If projection occurs, the offending velocity component is zeroed and angular
perturbation is damped:

.. math::

   \omega \leftarrow \gamma\,\omega,
   \quad \gamma = \texttt{brownian\_projection\_angular\_damping}

This reduces repeated wall-clipping oscillations.

Parameters in ``Simulation``
----------------------------

* ``brownian_longitudinal_std``
* ``brownian_transverse_std``
* ``brownian_rotation_std``
* ``brownian_persistence``
* ``brownian_application_mode``
* ``brownian_max_dx_fraction_of_trench_width``
* ``brownian_max_dy_fraction_of_segment_radius``
* ``brownian_max_dy_px_floor``
* ``brownian_max_dtheta``
* ``brownian_backoff_attempts``
* ``brownian_projection_angular_damping``

Behavioral Notes
----------------

* ``brownian_persistence = 0`` gives white-noise-like jitter.
* Higher persistence (e.g. ``0.8-0.95``) gives smoother drift.
* Setting all jitter std terms to zero disables Brownian perturbation.
* ``velocity`` / ``impulse`` generally look less repetitive than ``teleport``
  because contact resolution happens during physics stepping.
