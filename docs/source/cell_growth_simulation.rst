Bacterial growth simulations
============================

SyMBac contains two simulation backends. The first one uses the 2D physics library Pymunk_, and is the default simulator working with mother machine images. I am also in the process of adding experimental support for the CellModeller_ simulator, which is currently used for the 2D monolayer growth of cells on agar pads. In time I hope that CellModeller will entirely replace the Pymunk backend. 

Mother machine simulations
-----------------------------------

Mother machine simulations are handled with ``run_simulation``, which is in the ``phase_contrast_drawing`` module.

.. code-block:: python
   :caption: Running a simple simulation of cell growth in the mother machine.
   <options>

   from SyMBac.phase_contrast_drawing import run_simulation

   resize_amount = 3 #This is the "upscaling" factor for the simulation and the entire image generation process.
   pix_mic_conv = 0.0655 #e.g: 0.108379937 micron/pix for 60x, 0.0655 micron/pix for 100x
   scale = pix_mic_conv / resize_amount #The scale of the simulation
   sim_length = 100 #Number of timesteps to simulate


   cell_timeseries, space = run_simulation(
      trench_length=15, 
      trench_width=1.5, 
      cell_max_length=6, #6, long cells # 1.65 short cells
      cell_width= 1, #1 long cells # 0.95 short cells
      sim_length = sim_length,
      pix_mic_conv = pix_mic_conv,
      gravity=0,
      phys_iters=20,
      max_length_var = 3,
      width_var = 0.3,
      save_dir="/tmp/"
   )

.. _Pymunk: http://www.pymunk.org/en/latest/
.. _CellModeller: https://pubs.acs.org/doi/10.1021/sb300031n