Migration Guide: API Cleanup (v1.0)
===================================

This release introduces a hard API reset for simulation and rendering.

Quick Mapping
-------------

* ``Simulation(...many kwargs...)`` -> ``Simulation(SimulationSpec(...))``
* ``Simulation.run_simulation(...)`` -> ``Simulation.run(...)``
* ``Simulation.draw_simulation_OPL(...)`` -> ``Simulation.draw_opl(...)``
* ``SyMBac.cell_simulation.run_simulation(...)`` -> removed
* ``Renderer.generate_test_comparison(...)`` -> ``Renderer.render_frame(...)``
* ``Renderer.generate_training_data(...)`` -> ``Renderer.export_dataset(RandomDatasetPlan(...), ...)``
* ``Renderer.generate_timeseries_training_data(...)`` -> ``Renderer.export_dataset(TimeseriesDatasetPlan(...), ...)``
* ``renderer.params`` workflow -> ``RenderTuner.current_config()``
* ``SyMBac.auto_optimise.AutoOptimiser`` -> removed

Simulation Example
------------------

.. code-block:: python

   from SyMBac.config_models import (
       BrownianJitterSpec,
       SimulationCellSpec,
       SimulationGeometrySpec,
       SimulationPhysicsSpec,
       SimulationRuntimeSpec,
       SimulationSpec,
   )
   from SyMBac.simulation import Simulation

   spec = SimulationSpec(
       geometry=SimulationGeometrySpec(
           trench_length=15.0,
           trench_width=1.5,
           pix_mic_conv=0.065,
           resize_amount=3,
       ),
       cell=SimulationCellSpec(
           cell_max_length=6.5,
           cell_width=1.0,
           max_length_std=0.05,
           width_std=0.02,
           lysis_p=0.0,
       ),
       physics=SimulationPhysicsSpec(gravity=0.0, phys_iters=15),
       runtime=SimulationRuntimeSpec(sim_length=300, substeps=100, save_dir="/tmp/symbac_sim"),
       brownian=BrownianJitterSpec(application_mode="velocity"),
   )

   sim = Simulation(spec)
   sim.run(show_window=False)
   sim.draw_opl(do_transformation=False, label_masks=True)

Renderer + Export Example
-------------------------

.. code-block:: python

   from SyMBac.config_models import (
       DatasetOutputConfig,
       RenderConfig,
       TimeseriesDatasetPlan,
   )

   render_config = RenderConfig()
   plan = TimeseriesDatasetPlan(
       burn_in=40,
       sample_amount=0.02,
       n_series=3,
       frames_per_series=120,
   )
   output = DatasetOutputConfig(
       save_dir="/tmp/symbac_dataset",
       image_format="tiff",
       mask_dtype="uint16",
       export_geff=True,
   )

   metadata = renderer.export_dataset(
       plan=plan,
       output=output,
       base_config=render_config,
       seed=42,
   )

YAML Config Round-Trip
----------------------

All top-level config models support ``to_yaml``/``from_yaml``:

.. code-block:: python

   spec.to_yaml("simulation_spec.yaml")
   spec2 = SimulationSpec.from_yaml("simulation_spec.yaml")

   render_config.to_yaml("render_config.yaml")
   render_config2 = RenderConfig.from_yaml("render_config.yaml")
