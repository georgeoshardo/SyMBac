SyMBac.simulation
==========================

.. autoclass:: SyMBac.simulation.Simulation
    :members:

    .. automethod:: __init__

Compatibility Notes
-------------------

* ``max_length_var`` and ``width_var`` are deprecated aliases for ``max_length_std`` and ``width_std``.
  They emit a ``FutureWarning`` and are scheduled for removal on ``2026-09-01``.
* ``max_length_std`` and ``width_std`` are interpreted in simulation world units and converted
  using the full resolution ``scale_factor`` across the maintained simulation APIs.
* All parameters after ``cell_max_length`` in ``Simulation.__init__`` are keyword-only to prevent
  silent positional mis-assignment.
* ``substeps`` must be a positive integer.
* ``Simulation.run_simulation`` writes ``cell_timeseries.p`` and ``space_timeseries.p`` into ``save_dir``.
* ``Simulation(load_sim_dir=...)`` expects both artifacts in ``load_sim_dir``.
