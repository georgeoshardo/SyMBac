import logging
import os
import pickle
import random
import warnings
from copy import copy, deepcopy
from pathlib import Path

import napari
import numpy as np
import pyglet
import pymunk
from joblib import Parallel, delayed
from napari.utils.colormaps import label_colormap
from pymunk.pyglet_util import DrawOptions
from scipy.stats import norm
from tqdm.auto import tqdm

from symbac.cell import Cell, SimCell
from symbac.drawing import draw_scene, gen_cell_props_for_draw, generate_curve_props, get_space_size
from symbac.trench_geometry import get_trench_segments, trench_creator

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create file handler and set level to debug
log_path = os.path.join(os.getcwd(), "simulation.log")
fh = logging.FileHandler(log_path)
fh.setLevel(logging.DEBUG)

# Create formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(fh)


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

    def __init__(
        self,
        trench_length,
        trench_width,
        cell_max_length,
        max_length_var,
        cell_width,
        width_var,
        lysis_p,
        sim_length,
        pix_mic_conv,
        gravity,
        phys_iters,
        resize_amount,
        save_dir,
        load_sim_dir=None,
        sim_callback=None,
        show_progress="tqdm",
        random_seed=42,
    ):
        random.seed(random_seed)
        np.random.seed(random_seed)
        logger.info("Initializing Simulation object")

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
        self.load_sim_dir = load_sim_dir
        self.sim_callback = sim_callback
        self.show_progress = show_progress
        self.chronological_time = 0
        self.frame_time = 0

        try:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        except:
            pass

        if self.load_sim_dir:
            print(
                "Loading previous simulation, no need to call run_simulation method, but you still need to run OPL drawing and correctly define the scale"
            )
            with open(f"{load_sim_dir}/cell_timeseries.p", "rb") as f:
                self.cell_timeseries = pickle.load(f)
            with open(f"{load_sim_dir}/space_timeseries.p", "rb") as f:
                self.space = pickle.load(f)

        logger.debug(vars(self))

    def run_simulation(self, show_window=False):
        if show_window:
            warnings.warn(
                "You are using show_window = True. If you re-run the simulation (even by re-creating the Simulation object), then for reasons which I do not understand, the state of the simulation is not reset. Restart your notebook or interpreter to re-run simulations."
            )
        """
        Run the simulation

        :param bool show_window: Whether to show the pyglet window while running the simulation. Typically would be `false` if running SyMBac headless.

        """

        self.run_cell_simulation(
            show_window=show_window,
        )  # growth phase

    def draw_simulation_OPL(
        self, do_transformation=True, label_masks=True, return_output=False
    ):  # TODO decouble drawing from simulation
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
            delayed(gen_cell_props_for_draw)(a, ID_props)
            for a in tqdm(self.cell_timeseries, desc="Extracting cell properties from the simulation")
        )

        space_size = get_space_size(self.cell_timeseries_properties)

        scenes = Parallel(n_jobs=-1)(
            delayed(draw_scene)(cell_properties, do_transformation, space_size, self.offset, label_masks)
            for cell_properties in tqdm(self.cell_timeseries_properties, desc="Rendering cell optical path lengths")
        )
        self.OPL_scenes = [_[0] for _ in scenes]
        self.masks = [_[1] for _ in scenes]

        if return_output:
            return self.OPL_scenes, self.masks

    def visualise_in_napari(self):  # TODO decouble drawing from simulation
        """
        Opens a napari window allowing you to visualise the simulation, with both masks, OPL images, interactively.
        :return:
        """

        viewer = napari.view_image(np.array(self.OPL_scenes), name="OPL scenes")
        viewer.add_labels(np.array(self.masks), name="Synthetic masks")
        napari.run()

    def run_cell_simulation(
        self, show_window=True
    ):  # TODO make the dt and growth rate explicitly able to control the simulation step
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
        # space.iterations = 1000
        # space.damping = 0
        # space.collision_bias = 0.0017970074436457143*10
        self.space.collision_slop = 0.0
        self.dt = 1 / 20  # time-step per frame
        self.pix_mic_conv_for_sim = 1 / self.pix_mic_conv  # micron per pixel
        scale_factor = self.pix_mic_conv_for_sim * self.resize_amount  # resolution scaling factor

        self.trench_width_for_sim = self.trench_width * scale_factor
        self.trench_length_for_sim = self.trench_length * scale_factor - self.trench_width_for_sim / 2

        global_xy = (-self.trench_width_for_sim / 2, self.trench_width_for_sim / 2)
        trench_creator(self.trench_width_for_sim, self.trench_length_for_sim, global_xy, self.space)

        # Always set the N cells to 1 before adding a cell to the space, and set the mask_label
        self.space.historic_N_cells = 1
        cell1 = SimCell(
            length=self.cell_max_length * 0.5 * scale_factor,
            width=self.cell_width * scale_factor,
            resolution=20,
            position=(0, self.cell_max_length * scale_factor / 2),
            angle=np.pi / 2,
            growth_rate_constant=1,
            max_length=self.cell_max_length * scale_factor,
            max_length_mean=self.cell_max_length * scale_factor,
            max_length_var=self.max_length_var * np.sqrt(scale_factor),
            width_var=self.width_var * np.sqrt(scale_factor),
            width_mean=self.cell_width * scale_factor,
            mother=None,
            lysis_p=self.lysis_p,
            mask_label=1,
            generation=0,
            replicative_age=0,
            chronological_age=0,
            frame_age=0,
            simulation=self,
        )
        # for x in range(100):
        #    self.space.step(1/100)
        if show_window:
            import pyglet
            from pyglet.math import Mat4, Vec3

            window_height = 1000  # px
            self.__zoom = window_height / (self.trench_length_for_sim + self.trench_width_for_sim)
            self.__camera_offset = Vec3(700 / 2, 0, 0)
            self.__dragging = False
            window = pyglet.window.Window(
                width=700,
                height=1000,
                caption="SyMBac",
                resizable=True,
            )
            # window.view = window.view.from_translation(pyglet.math.Vec3(0, 0, 0))
            options = DrawOptions()
            options.shape_outline_color = (10, 20, 30, 40)

            @window.event
            def on_draw():
                window.clear()
                translation = Mat4.from_translation(self.__camera_offset)
                scaling = Mat4.from_scale(Vec3(self.__zoom, self.__zoom, 1.0))
                window.view = translation @ scaling
                self.space.debug_draw(options)

            @window.event
            def on_key_press(symbol, modifiers):
                if symbol == pyglet.window.key.LEFT:
                    self.__camera_offset += Vec3(10, 0, 0)
                elif symbol == pyglet.window.key.RIGHT:
                    self.__camera_offset -= Vec3(10, 0, 0)
                elif symbol == pyglet.window.key.UP:
                    self.__camera_offset -= Vec3(0, 10, 0)
                elif symbol == pyglet.window.key.DOWN:
                    self.__camera_offset += Vec3(0, 10, 0)
                elif symbol == pyglet.window.key.Z:
                    self.__zoom *= 1.1  # Zoom in
                elif symbol == pyglet.window.key.X:
                    self.__zoom /= 1.1  # Zoom out

            @window.event
            def on_mouse_press(x, y, button, modifiers):
                if button == pyglet.window.mouse.LEFT:
                    self.__dragging = True
                    self.__last_mouse_position = (x, y)

            @window.event
            def on_mouse_release(x, y, button, modifiers):
                if button == pyglet.window.mouse.LEFT:
                    self.__dragging = False

            @window.event
            def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
                if self.__dragging:
                    self.__camera_offset += Vec3(dx, dy, 0)
                    self.__last_mouse_position = (x, y)

            @window.event
            def on_mouse_scroll(x, y, scroll_x, scroll_y):
                zoom_factor = 1.1
                if scroll_y > 0:
                    self.__zoom *= zoom_factor  # Zoom in
                elif scroll_y < 0:
                    self.__zoom /= zoom_factor  # Zoom out

        x = [0]
        self.cell_timeseries = []
        self.cells = [cell1]
        self.historic_cells = [cell1]  # A list of cells which will contain all cells ever in the simulation
        self.sim_progress = 0
        self.progress_bar = tqdm(total=self.sim_length)

        if show_window:
            pyglet.clock.schedule_interval(self.step_and_update, interval=self.dt)
            pyglet.app.run()
        else:
            if self.show_progress == "magicgui":
                loop_iterator = self.pbar(range(self.sim_length + 2))
            else:
                loop_iterator = range(self.sim_length + 2)
            for _ in range(self.sim_length + 2):
                self.step_and_update(self.dt)
                if self.sim_callback:
                    self.sim_callback(self)

        return self.cell_timeseries, self.space, self.historic_cells

    def create_space(self):
        """
        Creates a pymunk space

        :return pymunk.Space space: A pymunk space
        """

        self.space = pymunk.Space(threaded=True)
        self.space.historic_N_cells = 0
        self.space.threads = 2

    def cell_adder(self, cell):
        cmap = (label_colormap(100).colors * 255).astype(int)
        cell.shape.color = cmap[cell.mask_label % len(cmap)]
        self.space.add(cell.body, cell.shape)

    def update_pm_cells(self):
        """
        Iterates through all cells in the simulation and updates their pymunk body and shape objects. Contains logic to
        check for cell division, and create daughters if necessary.

        :param list(SyMBac.cell.Cell) cells: A list of all cells in the current timepoint of the simulation.

        """
        for cell in self.cells:
            cell.update_length()
            if check_if_dividing(cell):
                daughter = cell.divide()
                daughter.mother = cell
                self.cells.append(daughter)
            else:
                cell.update_pm_cell()
            self.cell_adder(cell)
            for _ in range(150):
                self.space.step(1 / 100)

    def update_cell_positions(self):
        """
        Iterates through all cells in the simulation and updates their positions, keeping the cell object's position
        synchronised with its corresponding pymunk shape and body inside the pymunk space.

        :param list(SyMBac.cell.Cell) cells: A list of all cells in the current timepoint of the simulation.
        """
        for cell in self.cells:
            cell.update_position()

    def wipe_space(self):
        """
        Deletes all cells in the simulation pymunk space.

        :param pymunk.Space space:
        """
        for body, poly in zip(self.space.bodies, self.space.shapes):
            if body.body_type == 0:
                self.space.remove(body)
                self.space.remove(poly)

    def step_and_update(self, dt):  # dt dummy var in this case
        """
        The main simulation loop: Evolves the simulation forward

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
        print(self.sim_progress)

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
            elif norm.rvs() <= norm.ppf(cell.lysis_p) and len(self.cells) > 1:  # in case all cells disappear
                cell.dead = True
                self.cells.remove(cell)
                self.space.step(self.dt)
            else:
                pass
            self.historic_cells.append(cell)

        self.wipe_space()
        self.update_pm_cells()
        for _ in range(self.phys_iters):
            self.space.step(self.dt)
        self.update_cell_positions()

        if self.sim_progress > 1:
            self.cell_timeseries.append([cell.draw_cell_factory() for cell in self.cells])

        if self.sim_progress == self.sim_length:
            with open(self.save_dir + "/cell_timeseries.p", "wb") as f:
                pickle.dump(self.cell_timeseries, f)
            with open(self.save_dir + "/space_timeseries.p", "wb") as f:
                pickle.dump(self.space, f)
            pyglet.app.exit()
            print("Closed pyglet")
            return self.cells
        self.sim_progress += 1
        self.chronological_time += dt
        self.frame_time += 1
        self.progress_bar.update(1)
        return self.cells


def check_if_dividing(cell):
    if cell.length > cell.max_length:
        return True
    else:
        return False
