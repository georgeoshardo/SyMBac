SyMBac.simulation
==========================

.. autoclass:: SyMBac.simulation.Simulation
    :members:

    .. automethod:: __init__

Notes
-----

* ``Simulation`` now takes a single :class:`SyMBac.config_models.SimulationSpec`.
* Run via ``Simulation.run(...)`` and draw OPL via ``Simulation.draw_opl(...)``.
* Brownian controls live under ``SimulationSpec.brownian``. See
  :doc:`brownian_jitter_model` for equations and parameter guidance.
* ``Simulation.run`` writes ``cell_timeseries.p`` and ``space_timeseries.p`` into
  ``SimulationSpec.runtime.save_dir``.
* Loading from previous output is configured via
  ``SimulationSpec.runtime.load_sim_dir``.
* See :doc:`migration_v1_api_cleanup` for old-to-new API mapping.
