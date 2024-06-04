import numpy as np
from joblib import Parallel, delayed
import pickle
from SyMBac.drawing import draw_scene, get_space_size, gen_cell_props_for_draw, generate_curve_props
from SyMBac.trench_geometry import  get_trench_segments
import napari
import os
import warnings
from tqdm.auto import tqdm
from SyMBac.cell import Cell
from pymunk.pyglet_util import DrawOptions
import pymunk
import pyglet
from scipy.stats import norm
from copy import deepcopy
from SyMBac.trench_geometry import trench_creator, get_trench_segments

class Simulation:
    
    
    """
    Class for instantiating Simulation objects. These are the basic objects used to run all SyMBac simulations. This
    class is used to parameterise simulations, run them, draw optical path length images, and then visualise them.

    Example:

    >>> from SyMBac.simulation import Simulation
    >>> my_simulation = Simulation(
            trench_length=15,
            trench_width=1.3,
            cell_max_length=6.65, #6, long cells # 1.65 short cells
            cell_width= 1, #1 long cells # 0.95 short cells
            sim_length = 100,
            pix_mic_conv = 0.065,
            gravity=0,
            phys_iters=15,
            max_length_var = 0.,
            width_var = 0.,
            lysis_p = 0.,
            save_dir="/tmp/",
            resize_amount = 3
        )
    >>> my_simulation.run_simulation(show_window=False)
    >>> my_simulation.draw_simulation_OPL(do_transformation=True, label_masks=True)
    >>> my_simulation.visualise_in_napari()

    """

    def __init__(self, trench_length, trench_width, cell_max_length, max_length_var, cell_width, width_var, lysis_p, sim_length, pix_mic_conv, gravity, phys_iters, resize_amount, save_dir, load_sim_dir = None):
        """
        Initialising a Simulation object

        Parameters
        ----------
        trench_length : float
            Length of a mother machine trench (micron)
        trench_width : float
            Width of a mother machine trench (micron)
        cell_max_length : float
            Maximum length a cell can reach before dividing (micron)
        cell_width : float
            the average cell width in the simulation (micron)
        pix_mic_conv : float
            The micron/pixel size of the image
        gravity : float
            Pressure forcing cells into the trench. Typically left at zero, but can be varied if cells start to fall into
            each other or if the simulation behaves strangely.
        phys_iters : int
            Number of physics iterations per simulation frame. Increase to resolve collisions if cells are falling into one
            another, but decrease if cells begin to repel one another too much (too high a value causes cells to bounce off
            each other very hard). 20 is a good starting point
        max_length_var : float
            Variance of the maximum cell length
        width_var : float
            Variance of the maximum cell width
        save_dir : str
            Location to save simulation output
        lysis_p : float
            probability of cell lysis
        sim_length : int
            number of frames to simulate (where each one is dt). Start with 200-1000, and grow your simulation from there.
        resize_amount : int
            This is the "upscaling" factor for the simulation and the entire image generation process. Must be kept constant
            across the image generation pipeline. Starting value of 3 recommended.
        load_sim_dir : str
            The directory if you wish to load a previously completed simulation
        """
        self.trench_length = trench_length
        self.trench_width = trench_width
        self.cell_max_length = cell_max_length
        self.max_length_var = max_length_var
        self.cell_width = cell_width
        self.width_var = width_var
        self.lysis_p = lysis_p
        self.sim_length = sim_length
        self.pix_mic_conv = pix_mic_conv
        self.gravity = gravity
        self.phys_iters = phys_iters
        self.resize_amount = resize_amount
        self.save_dir = save_dir
        self.offset = 30
        self.load_sim_dir = load_sim_dir

        try:
            os.mkdir(save_dir)
        except:
            pass

        if self.load_sim_dir:
            print("Loading previous simulation, no need to call run_simulation method, but you still need to run OPL drawing and correctly define the scale")
            with open(f"{load_sim_dir}/cell_timeseries.p", 'rb') as f:
                self.cell_timeseries = pickle.load(f)
            with open(f"{load_sim_dir}/space_timeseries.p", 'rb') as f:
                self.space = pickle.load(f)

    


    def run_simulation(self, show_window = True):
        if show_window:
            warnings.warn("You are using show_window = True. If you re-run the simulation (even by re-creating the Simulation object), then for reasons which I do not understand, the state of the simulation is not reset. Restart your notebook or interpreter to re-run simulations.")
        """
        Run the simulation

        :param bool show_window: Whether to show the pyglet window while running the simulation. Typically would be `false` if running SyMBac headless.

        """

        self.run_cell_simulation(
            show_window = show_window,
        )  # growth phase

    def draw_simulation_OPL(self, do_transformation = True, label_masks = True, return_output = False):


        """
        Draw the optical path length images from the simulation. This involves drawing the 3D cells into a 2D numpy
        array, and then the corresponding masks for each cell.

        After running this function, the Simulation object will gain two new attributes: ``self.OPL_scenes`` and ``self.masks`` which can be accessed separately.

        :param bool do_transformation: Sets whether to transform the cells by bending them. Bending the cells can add realism to a simulation, but risks clipping the cells into the mother machine trench walls.

        :param bool label_masks: Sets whether the masks should be binary, or labelled. Masks should be binary is training a standard U-net, such as with DeLTA, but if training Omnipose (recommended), then mask labelling should be set to True.

        :param bool return_output: Controls whether the function returns the OPL scenes and masks. Does not affect the assignment of these attributes to the instance.

        Returns
        -------
        output : tuple(list(numpy.ndarray), list(numpy.ndarray))
           If ``return_output = True``, a tuple containing lists, each of which contains the entire simulation. The first element in the tuple contains the OPL images, the second element contains the masks

        """
        self.main_segments = get_trench_segments(self.space)
        ID_props = generate_curve_props(self.cell_timeseries)

        self.cell_timeseries_properties = Parallel(n_jobs=-1)(
            delayed(gen_cell_props_for_draw)(a, ID_props) for a in tqdm(self.cell_timeseries, desc='Extracting cell properties from the simulation'))

        space_size = get_space_size(self.cell_timeseries_properties)

        scenes = Parallel(n_jobs=-1)(delayed(draw_scene)(
        cell_properties, do_transformation, space_size, self.offset, label_masks) for cell_properties in tqdm(
            self.cell_timeseries_properties, desc='Rendering cell optical path lengths'))
        self.OPL_scenes = [_[0] for _ in scenes]
        self.masks = [_[1] for _ in scenes]

        if return_output:
            return self.OPL_scenes, self.masks

    def visualise_in_napari(self):
        """
        Opens a napari window allowing you to visualise the simulation, with both masks, OPL images, interactively.
        :return:
        """
        
        viewer = napari.view_image(np.array(self.OPL_scenes), name='OPL scenes')
        viewer.add_labels(np.array(self.masks), name='Synthetic masks')
        napari.run()


    def run_cell_simulation(self, show_window = True):
        """
        Runs the rigid body simulation of bacterial growth based on a variety of parameters. Opens up a Pyglet window to
        display the animation in real-time. If the simulation looks bad to your eye, restart the kernel and rerun the
        simulation. There is currently a bug where if you try to rerun the simulation in the same kernel with show_window=True, it will be
        extremely slow.

        Parameters
        ----------

        trench_length : float
            Length of a mother machine trench (micron)
        trench_width : float
            Width of a mother machine trench (micron)
        cell_max_length : float
            Maximum length a cell can reach before dividing (micron)
        cell_width : float
            the average cell width in the simulation (micron)
        pix_mic_conv : float
            The micron/pixel size of the image
        gravity : float
            Pressure forcing cells into the trench. Typically left at zero, but can be varied if cells start to fall into
            each other or if the simulation behaves strangely.
        phys_iters : int
            Number of physics iterations per simulation frame. Increase to resolve collisions if cells are falling into one
            another, but decrease if cells begin to repel one another too much (too high a value causes cells to bounce off
            each other very hard). 20 is a good starting point
        max_length_var : float
            Variance of the maximum cell length
        width_var : float
            Variance of the maximum cell width
        save_dir : str
            Location to save simulation output
        lysis_p : float
            probability of cell lysis

        Returns
        -------
        cell_timeseries : lists
            A list of parameters for each cell, such as length, width, position, angle, etc. All used in the drawing of the
            scene later
        space : a pymunk space object
            Contains the rigid body physics objects which are the cells.
        """

        self.create_space()
        self.space.gravity = 0, self.gravity  # arbitrary units, negative is toward trench pole
        #space.iterations = 1000
        #space.damping = 0
        #space.collision_bias = 0.0017970074436457143*10
        self.space.collision_slop = 0.
        self.dt = 1 / 20  # time-step per frame
        self.pix_mic_conv_for_sim = 1 / self.pix_mic_conv  # micron per pixel
        scale_factor = self.pix_mic_conv_for_sim * self.resize_amount  # resolution scaling factor

        self.trench_length_for_sim = self.trench_length * scale_factor
        self.trench_width_for_sim = self.trench_width * scale_factor
        trench_creator(self.trench_width_for_sim, self.trench_length_for_sim, (35, 0), self.space)  # Coordinates of bottom left corner of the trench

        # Always set the N cells to 1 before adding a cell to the space, and set the mask_label
        self.space.historic_N_cells = 1
        cell1 = Cell(
            length=self.cell_max_length*0.5 * scale_factor,
            width=self.cell_width * scale_factor,
            resolution=60,
            position=(40, 100),
            angle=0.8,
            space=self.space,
            dt= self.dt,
            growth_rate_constant=1,
            max_length=self.cell_max_length * scale_factor,
            max_length_mean=self.cell_max_length * scale_factor,
            max_length_var=self.max_length_var * np.sqrt(scale_factor),
            width_var=self.width_var * np.sqrt(scale_factor),
            width_mean=self.cell_width * scale_factor,
            mother=None,
            lysis_p=self.lysis_p,
            mask_label=1,
            generation = 0,
            N_divisions=0
        )

        if show_window:

            window = pyglet.window.Window(700, 700, "SyMBac", resizable=True)
            options = DrawOptions()
            options.shape_outline_color = (10,20,30,40)
            @window.event
            def on_draw():
                window.clear()
                self.space.debug_draw(options)

            # key press event
            @window.event
            def on_key_press(symbol, modifier):

                # key "E" get press
                if symbol == pyglet.window.key.E:
                    # close the window
                    window.close()

        x = [0]
        self.cell_timeseries = []
        self.cells = [cell1]
        self.historic_cells = [cell1] # A list of cells which will contain all cells ever in the simulation
        self.sim_progress = 0
        if show_window:
            pyglet.clock.schedule_interval(self.step_and_update, interval = self.dt)
            pyglet.app.run()
        else:
            for _ in tqdm(range(self.sim_length+2)):
                self.step_and_update(self.dt)


        for frame, cells in enumerate(self.cell_timeseries):
            for cell in cells:
                cell.t = frame#

        return self.cell_timeseries, self.space, self.historic_cells

    def create_space(self):
        """
        Creates a pymunk space

        :return pymunk.Space space: A pymunk space
        """

        self.space = pymunk.Space(threaded=True)
        self.space.historic_N_cells = 0
        #space.threads = 2



    def update_pm_cells(self, cells, space):
        """
        Iterates through all cells in the simulation and updates their pymunk body and shape objects. Contains logic to
        check for cell division, and create daughters if necessary.

        :param list(SyMBac.cell.Cell) cells: A list of all cells in the current timepoint of the simulation.

        """
        for cell in cells:
            cell.update_length()
            if cell.is_dividing():
                daughter_details = cell.create_pm_cell()
                if len(daughter_details) > 2: # Really hacky. Needs fixing because sometimes this returns cell_body, cell shape. So this is a check to ensure that it's returing daughter_x, y and angle
                    daughter = Cell(**daughter_details)
                    cell.daughter = daughter
                    daughter.mother = cell
                    #daughter.mo
                    cells.append(daughter)
            else:
                cell.create_pm_cell()
            self.cell_adder(cell, space)
            for _ in range(150):
                space.step(1/100)

    def cell_adder(self, cell, space):
        space.add(cell.body, cell.shape)

    def update_cell_positions(self, cells):
        """
        Iterates through all cells in the simulation and updates their positions, keeping the cell object's position
        synchronised with its corresponding pymunk shape and body inside the pymunk space.

        :param list(SyMBac.cell.Cell) cells: A list of all cells in the current timepoint of the simulation.
        """
        for cell in cells:
            cell.update_position()

    def wipe_space(self, space):
        """
        Deletes all cells in the simulation pymunk space.

        :param pymunk.Space space:
        """
        for body, poly in zip(space.bodies, space.shapes):
            if body.body_type == 0:
                space.remove(body)
                space.remove(poly)


    def step_and_update(self, dt): #dt dummy var in this case
        """
        Evolves the simulation forward

        :param float dt: The simulation timestep
        :param list(SyMBac.cell.Cell)  cells: A list of all cells in the current timestep
        :param pymunk.Space space: The simulations's pymunk space.
        :param int phys_iters: The number of physics iteration in each timestep
        :param int ylim: The y coordinate threshold beyond which to delete cells
        :param list cell_timeseries: A list to store the cell's properties each time the simulation steps forward
        :param int list: A list with a single value to store the simulation's progress.
        :param int sim_length: The number of timesteps to run.
        :param str save_dir: The directory to save the simulation information.

        Returns
        -------
        cells : list(SyMBac.cell.Cell)

        """
        for shape in self.space.shapes:
            if shape.body.position.y < 0 or shape.body.position.y > self.trench_length_for_sim:
                self.space.remove(shape.body, shape)
                self.space.step(self.dt)

        for cell in self.cells:
            self.historic_cells.append(cell)
            if cell.shape.body.position.y < 0 or cell.shape.body.position.y > self.trench_length_for_sim:
                cell.dead = True
                self.cells.remove(cell)
                self.space.step(self.dt)
            elif norm.rvs() <= norm.ppf(cell.lysis_p) and len(self.cells) > 1:   # in case all cells disappear
                cell.dead = True
                self.cells.remove(cell)
                self.space.step(self.dt)
            else:
                pass
            self.historic_cells.append(cell)


        self.wipe_space(self.space)
        self.update_pm_cells(self.cells, self.space)
        for _ in range(self.phys_iters):
            self.space.step(self.dt)
        self.update_cell_positions(self.cells)

        if self.sim_progress > 1:
            self.cell_timeseries.append(deepcopy(self.cells))
        if self.sim_progress == self.sim_length-1:
            with open(self.save_dir+"/cell_timeseries.p", "wb") as f:
                pickle.dump(self.cell_timeseries, f)
            with open(self.save_dir+"/space_timeseries.p", "wb") as f:
                pickle.dump(self.space, f)
            pyglet.app.exit()
            return self.cells
        self.sim_progress += 1
        return self.cells

