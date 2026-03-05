SyMBac in Napari
================

SyMBac now includes a structured napari workflow with five docks:

* **SyMBac Simulation**: edit/load/save ``SimulationSpec`` YAML, run simulation, draw OPL scenes.
* **SyMBac Optics**: pick a real image, configure PSF/camera, build renderer.
* **SyMBac Regions**: bootstrap or manually edit media/cell/device region labels and apply them to rendering.
* **SyMBac Tuning**: tune ``RenderConfig`` values and preview synthetic frames.
* **SyMBac Export**: export random datasets, timeseries datasets, or batch timeseries variants.

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

YAML Persistence
----------------

The napari docks support explicit YAML load/save for:

* ``SimulationSpec``
* ``RenderConfig``
* ``RandomDatasetPlan`` / ``TimeseriesDatasetPlan``
* ``DatasetOutputConfig``
