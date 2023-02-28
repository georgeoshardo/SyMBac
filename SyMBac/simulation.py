import numpy as np
from joblib import Parallel, delayed

from SyMBac.cell_simulation import run_simulation
from SyMBac.drawing import draw_scene, get_space_size, gen_cell_props_for_draw, generate_curve_props
from SyMBac.trench_geometry import  get_trench_segments
from tqdm.autonotebook import tqdm
import napari

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

    def __init__(self, trench_length, trench_width, cell_max_length, max_length_var, cell_width, width_var, lysis_p, sim_length, pix_mic_conv, gravity, phys_iters, resize_amount, save_dir, objective):
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
        objective : int
            The objective magnification of the images you are trying to replicate
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
        self.objective = objective

    def run_simulation(self, show_window = True):
        """
        Run the simulation

        :param bool show_window: Whether to show the pyglet window while running the simulation. Typically would be `false` if running SyMBac headless.

        """
        self.cell_timeseries, self.space = run_simulation(
            trench_length=self.trench_length,
            trench_width=self.trench_width,
            cell_max_length=self.cell_max_length,  # 6, long cells # 1.65 short cells
            cell_width=self.cell_width,  # 1 long cells # 0.95 short cells
            sim_length=self.sim_length,
            pix_mic_conv=self.pix_mic_conv,
            gravity=self.gravity,
            phys_iters=self.phys_iters,
            max_length_var=self.max_length_var,
            width_var=self.width_var,
            lysis_p=self.lysis_p,  # this should somehow depends on the time
            save_dir=self.save_dir,
            show_window = show_window
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
            delayed(gen_cell_props_for_draw)(a, ID_props) for a in tqdm(self.cell_timeseries, desc='Timeseries Properties'))

        space_size = get_space_size(self.cell_timeseries_properties)

        scenes = Parallel(n_jobs=-1)(delayed(draw_scene)(
        cell_properties, do_transformation, space_size, self.offset, label_masks) for cell_properties in tqdm(
            self.cell_timeseries_properties, desc='Scene Draw:'))
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