Bacterial growth simulations
============================

SyMBac contains two simulation backends. The first one uses the 2D physics library Pymunk_, and is the default simulator working with mother machine images. I am also in the process of adding experimental support for the CellModeller_ simulator, which is currently used for the 2D monolayer growth of cells on agar pads. In time I hope that CellModeller will entirely replace the Pymunk backend. 

Mother machine simulations
-----------------------------------

Mother machine simulations are handled with ``run_simulation``, which is in the ``phase_contrast_drawing`` module.

``run_simulation`` takes various arguments which need to be specified, all of which are in micrometers unless otherwise specified:

- *trench_length*: The length of the trench.
- *trench_width*: The width of the trench.
- *cell_max_length*: The maximum allowable length of a cell in the simulation.
- *cell_width*: The average width of cells in the simulation.
- *sim_length*: The number of timesteps to run the simulation for.
- *pix_mic_conv*: The number of microns per pixel in the simulation. 
- *gravity*: The strength of the arbitrary gravity force in the simulations.
- *phys_iters*: The number of iterations of the rigid body physics solver to run each timestep. Note that this affects how gravity works in the simulation, as gravity is applied every physics iteration, higher values of *phys_iters* will result in more gravity if it is turned on.
- *max_length_var*: The variance applied to the normal distribution which has mean *cell_max_length*, from which maximum cell lengths are sampled.
- *width_var*: The variance applied to the normal distribution of cell widths which has mean *cell_width*.
- *save_dir*: The save location of the return value of the function. The output will be pickled and saved here, so that the simulation can be reloaded later withuot having to rerun it, for reproducibility. If you don't want to save it, just leave it as ``/tmp/``.

.. code-block:: python
   :caption: Running a simple simulation of cell growth in the mother machine.

   from SyMBac.phase_contrast_drawing import run_simulation, get_trench_segments

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


The ``run_simulation`` function returns two objects. The latter is a ``pymunk.space.Space`` object, and typically does not need to be accessed. The first object returned is ``cell_timeseries``. This is a list of lists, with each sub-list contining a ``SyMBac.cell.Cell`` object. These are the individual cells in the simulation, at each timepoint. For instance, ``cell_timeseries[0]`` contains all the cells at timepoint 0, and ``cell_timeseries[0][0]`` is one of the cells in this frame. ``SyMBac.cell.Cell`` have many cellular attributes which can be accessed, with a full description in the API.

We will now proceed to generate the the scenes, which contain the raw unconvolved images of the cells. The following code block extracts the precise position of the trench from the simulation into a variable called ``main_segments``. 

The next step is identifying all unique cells within the simulation and assigning them 4 properties which will later be used to morph and bend the cells to make them look more realistic. This is described in more detail in the documentation for ``generate_curve_props``. 

.. code-block:: python
   :caption: Generating the scenes from the simulation.

    main_segments = get_trench_segments(space) #Returns a pandas dataframe contianing information about the trench position and geometry for drawing.
    ID_props = generate_curve_props(cell_timeseries)

Next we will collate all the properties for each cell in each timepoint into one variable called ``cell_timeseries_properties``. This is achievd by calling the ``gen_cell_props_for_draw``, and passing in each ``SyMBac.cell.Cell`` object, aloing with the ``ID_props`` variable from above. It will return, for each cell, the length, width, angle, centroid, freq_modif, amp_modif, phase_modif, phase_mult (the last 4 being the ID props of each unique cell). This data allows us to redraw every cell onto a NumPy array and render an unconvolved raw image.

.. code-block:: python
   :caption: Generating the cell properties to draw the scene.

    from joblib import Parallel, delayed
    from tqdm.notebook import tqdm
    from SyMBac.general_drawing import generate_curve_props, gen_cell_props_for_draw, get_space_size, convolve_rescale
    from SyMbac.general

    cell_timeseries_properties = Parallel(n_jobs=-1)(
        delayed(gen_cell_props_for_draw)(a, ID_props) for a in tqdm(cell_timeseries, desc='Timeseries Properties'))


.. _Pymunk: http://www.pymunk.org/en/latest/
.. _CellModeller: https://pubs.acs.org/doi/10.1021/sb300031n