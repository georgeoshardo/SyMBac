SyMBac in Napari
================

SyMBac now includes a structured napari workflow with one main dock and five tabs:

* **Simulation**: set simulation parameters, run simulation, draw OPL scenes.
* **Optics**: pick a real image, configure PSF/camera, build renderer.
* **Regions**: bootstrap or manually edit media/cell/device region labels and apply them to rendering.
* **Tuning**: tune ``RenderConfig`` values and preview synthetic frames.
* **Export**: export random datasets, timeseries datasets, or batch timeseries variants.

Launch
------

From Python:

.. code-block:: python

   from SyMBac import launch_napari
   launch_napari()

From terminal:

.. code-block:: bash

   symbac-napari

Notebook Parity
---------------

The workflow mirrors the two reference examples:

* ``examples/simple_mother_machine.ipynb``
* ``examples/timeseries_tracking_data.ipynb``

including simulation setup, OPL drawing, renderer setup, region-label-guided tuning, and dataset export.

YAML Persistence (Advanced)
---------------------------

The workflow is form-first by default. Advanced YAML panels can be expanded when needed.

The advanced panels support explicit YAML load/save for:

* ``SimulationSpec``
* ``RenderConfig``
* ``RandomDatasetPlan`` / ``TimeseriesDatasetPlan``
* ``DatasetOutputConfig``
