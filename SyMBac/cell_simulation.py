import pickle
from copy import deepcopy
import numpy as np
from scipy.stats import norm
from SyMBac.cell import Cell
from SyMBac.trench_geometry import trench_creator, get_trench_segments
from pymunk.pyglet_util import DrawOptions
import pymunk
import pyglet
from tqdm.autonotebook import tqdm

def run_simulation(trench_length, trench_width, cell_max_length, cell_width, sim_length, pix_mic_conv, gravity,
                   phys_iters, max_length_var, width_var, save_dir, lysis_p=0, show_window = True):
    """
    Runs the rigid body simulation of bacterial growth based on a variety of parameters. Opens up a Pyglet window to
    display the animation in real-time. If the simulation looks bad to your eye, restart the kernel and rerun the
    simulation. There is currently a bug where if you try to rerun the simulation in the same kernel, it will be
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

    space = create_space()
    space.gravity = 0, gravity  # arbitrary units, negative is toward trench pole
    #space.iterations = 1000
    #space.damping = 0
    #space.collision_bias = 0.0017970074436457143*10
    space.collision_slop = 0.
    dt = 1 / 20  # time-step per frame
    pix_mic_conv = 1 / pix_mic_conv  # micron per pixel
    scale_factor = pix_mic_conv * 3  # resolution scaling factor

    trench_length = trench_length * scale_factor
    trench_width = trench_width * scale_factor
    trench_creator(trench_width, trench_length, (35, 0), space)  # Coordinates of bottom left corner of the trench

    cell1 = Cell(
        length=cell_max_length * scale_factor,
        width=cell_width * scale_factor,
        resolution=60,
        position=(20 + 35, 10),
        angle=0.8,
        space=space,
        dt= dt,
        growth_rate_constant=1,
        max_length=cell_max_length * scale_factor,
        max_length_mean=cell_max_length * scale_factor,
        max_length_var=max_length_var * np.sqrt(scale_factor),
        width_var=width_var * np.sqrt(scale_factor),
        width_mean=cell_width * scale_factor,
        parent=None,
        lysis_p=lysis_p
    )

    if show_window:

        window = pyglet.window.Window(700, 700, "SyMBac", resizable=True)
        options = DrawOptions()
        options.shape_outline_color = (10,20,30,40)
        @window.event
        def on_draw():
            window.clear()
            space.debug_draw(options)

        # key press event
        @window.event
        def on_key_press(symbol, modifier):

            # key "E" get press
            if symbol == pyglet.window.key.E:
                # close the window
                window.close()

    #global cell_timeseries
    #global x

    #try:
    #    del cell_timeseries
    #except:
    #    pass
    #try:
    #    del x
    #except:
    #    pass

    x = [0]
    cell_timeseries = []
    cells = [cell1]
    if show_window:
        pyglet.clock.schedule_interval(step_and_update, interval=dt, cells=cells, space=space, phys_iters=phys_iters,
                                       ylim=trench_length, cell_timeseries=cell_timeseries, x=x, sim_length=sim_length,
                                       save_dir=save_dir)
        pyglet.app.run()
    else:
        for _ in tqdm(range(sim_length)):
            step_and_update(
                dt=dt, cells=cells, space=space, phys_iters=phys_iters, ylim=trench_length,
                cell_timeseries=cell_timeseries, x=x, sim_length=sim_length, save_dir=save_dir
            )

    # window.close()
    # phys_iters = phys_iters
    # for x in tqdm(range(sim_length+250),desc="Simulation Progress"):
    #    cells = step_and_update(dt=dt, cells=cells, space=space, phys_iters=phys_iters,ylim=trench_length*1.1, cell_timeseries = cell_timeseries, x=x, sim_length = sim_length, save_dir = save_dir)
    #    if x > 250:
    #        cell_timeseries.append(deepcopy(cells))
    return cell_timeseries, space

def create_space():
    """
    Creates a pymunk space

    :return pymunk.Space space: A pymunk space
    """

    space = pymunk.Space(threaded=False)
    #space.threads = 2
    return space

def update_cell_lengths(cells):
    """
    Iterates through all cells in the simulation and updates their length according to their growth law.

    :param list(SyMBac.cell.Cell) cells: A list of all cells in the current timepoint of the simulation.
    """
    for cell in cells:
        cell.update_length()


def update_pm_cells(cells):
    """
    Iterates through all cells in the simulation and updates their pymunk body and shape objects. Contains logic to
    check for cell division, and create daughters if necessary.

    :param list(SyMBac.cell.Cell) cells: A list of all cells in the current timepoint of the simulation.

    """
    for cell in cells:
        if cell.is_dividing():
            daughter_details = cell.create_pm_cell()
            if len(daughter_details) > 2: # Really hacky. Needs fixing because sometimes this returns cell_body, cell shape. So this is a check to ensure that it's returing daughter_x, y and angle
                daughter = Cell(**daughter_details)
                cell.daughter = daughter
                cells.append(daughter)
        else:
            cell.create_pm_cell()

def update_cell_positions(cells):
    """
    Iterates through all cells in the simulation and updates their positions, keeping the cell object's position
    synchronised with its corresponding pymunk shape and body inside the pymunk space.

    :param list(SyMBac.cell.Cell) cells: A list of all cells in the current timepoint of the simulation.
    """
    for cell in cells:
        cell.update_position()

def wipe_space(space):
    """
    Deletes all cells in the simulation pymunk space.

    :param pymunk.Space space:
    """
    for body, poly in zip(space.bodies, space.shapes):
        if body.body_type == 0:
            space.remove(body)
            space.remove(poly)

def update_cell_parents(cells, new_cells):
    """
    Takes two lists of cells, one in the previous frame, and one in the frame after division, and updates the parents of
    each cell

    :param list(SyMBac.cell.Cell) cells:
    :param list(SyMBac.cell.Cell) new_cells:
    """
    for i in range(len(cells)):
        cells[i].update_parent(id(new_cells[i]))

def step_and_update(dt, cells, space, phys_iters, ylim, cell_timeseries,x,sim_length,save_dir):
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
    for shape in space.shapes:
        if shape.body.position.y < 0 or shape.body.position.y > ylim:
            space.remove(shape.body, shape)
            space.step(dt)
    #new_cells = []
    #graveyard = []
    for cell in cells:
        if cell.shape.body.position.y < 0 or cell.shape.body.position.y > ylim:
            #graveyard.append([cell, "outside"])
            cells.remove(cell)
            space.step(dt)
        elif norm.rvs() <= norm.ppf(cell.lysis_p) and len(cells) > 1:   # in case all cells disappear
            #graveyard.append([cell, "lysis"])
            cells.remove(cell)
            space.step(dt)
        else:
            pass
            #new_cells.append(cell)
    #cells = deepcopy(new_cells)
    #graveyard = deepcopy(graveyard)

    wipe_space(space)

    update_cell_lengths(cells)
    update_pm_cells(cells)

    for _ in range(phys_iters):
        space.step(dt)
    update_cell_positions(cells)

    #print(str(len(cells))+" cells")
    if x[0] > 1:
        #copy_cells = deepcopy(cells)

        cell_timeseries.append(deepcopy(cells))
        copy_cells = cell_timeseries[-1]
        update_cell_parents(cells, copy_cells)
        #del copy_cells
    if x[0] == sim_length-1:
        with open(save_dir+"/cell_timeseries.p", "wb") as f:
            pickle.dump(cell_timeseries, f)
        with open(save_dir+"/space_timeseries.p", "wb") as f:
            pickle.dump(space, f)
        pyglet.app.exit()
        return cells
    x[0] += 1
    return (cells)

