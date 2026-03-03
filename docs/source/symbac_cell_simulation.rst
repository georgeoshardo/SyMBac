SyMBac.cell_simulation
==========================

.. automodule:: SyMBac.cell_simulation
    :members:

Persistence Notes
-----------------

* ``run_simulation`` writes ``cell_timeseries.p`` and ``space_timeseries.p`` into ``save_dir``.
* ``SyMBac.simulation.Simulation(load_sim_dir=...)`` expects these artifacts in ``load_sim_dir``.
* ``max_length_std`` and ``width_std`` are converted with the full resolution ``scale_factor``
  in the legacy rigid-body implementation.
