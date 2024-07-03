import itertools
import random
from noise import pnoise2
import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from skimage.measure import label
from skimage.transform import rescale, rotate
from skimage.morphology import opening, remove_small_objects
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries
from PIL import Image
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from SyMBac.trench_geometry import get_trench_segments

div_odd = lambda n: (n // 2, n // 2 + 1)
perc_diff = lambda a, b: (a - b) / b * 100

def generate_curve_props(cell_timeseries):
    """
    Generates individual cell curvature properties. 3 parameters for each cell, which are passed to a cosine function to modulate the cell's curvature. 
    
    Parameters
    ---------
    cell_timeseries : list(cell_properties)
        The output of :meth:`SyMBac.simulation.Simulation.run_simulation()`
    
    Returns
    -------
    outupt : A numpy array of unique curvature properties for each cell in the simulation
    """

    # Get unique cell IDs
    IDs = []
    for cell_list in cell_timeseries:
        for cell in cell_list:
            IDs.append(cell.mask_label)
    IDs = np.array(IDs)
    unique_IDs = np.unique(IDs)
    # For each cell, assign random curvature properties
    ID_props = []
    for ID in unique_IDs:
        freq_modif = (np.random.uniform(0.9, 1.1))  # Choose one per cell
        amp_modif = (np.random.uniform(0.9, 1.1))  # Choose one per cell
        phase_modif = np.random.uniform(-1, 1)  # Choose one per cell
        ID_props.append([int(ID), freq_modif, amp_modif, phase_modif])
    ID_propps = np.array(ID_props)
    ID_propps[:, 0] = ID_propps[:, 0].astype(int)
    return np.array(ID_props)


def gen_cell_props_for_draw(cell_timeseries_lists, ID_props):
    """
    Parameters
    ----------
    cell_timeseries_lists : list
        A list (single frame) from cell_timeseries, the output from run_simulation. E.g: cell_timeseries[x]
    ID_props : list
        A list of properties for each cell in that frame, the output of generate_curve_props()
    
    Returns
    -------
    cell_properties : list
        The final property list used to actually draw a scene of cells. The input to draw_scene
    """

    cell_properties = []
    for cell in cell_timeseries_lists:
        body, shape = (cell.body, cell.shape)
        vertices = []
        for v in shape.get_vertices():
            x, y = v.rotated(shape.body.angle) + shape.body.position  # .rotated(self.shape.body.angle)
            vertices.append((x, y))
        vertices = np.array(vertices)

        centroid = get_centroid(vertices)
        farthest_vertices = find_farthest_vertices(vertices)
        length = get_distance(farthest_vertices[0], farthest_vertices[1])
        length = cell.length
        width = cell.width
        separation = cell.pinching_sep
        # angle = np.arctan(vertices_slope(farthest_vertices[0], farthest_vertices[1]))
        #angle = np.arctan2((farthest_vertices[0] - farthest_vertices[1])[1],
        #                   (farthest_vertices[0] - farthest_vertices[1])[0])
        angle = cell.angle #np.rad2deg(angle) + 90

        ID, freq_modif, amp_modif, phase_modif = ID_props[ID_props[:, 0] == cell.mask_label][0]
        phase_mult = 20
        draw_params = {
            "length": length,
            "width": width,
            "angle": angle,
            "centroid": centroid,
            "freq_modif": freq_modif,
            "amp_modif": amp_modif,
            "phase_modif": phase_modif,
            "phase_mult": phase_mult,
            "separation": separation,
            "mask_label": cell.mask_label
        }
        cell.draw_params = draw_params
        cell_properties.append([length, width, angle, centroid, freq_modif, amp_modif, phase_modif, phase_mult,
                                separation, cell.mask_label, cell.texture_y_coordinate])
    return cell_properties

def get_crop_bounds_2D(img, tol=0):
    mask = img>tol
    x_idx = np.ix_(mask.any(1),mask.any(0))
    start_row, stop_row, start_col, stop_col = x_idx[0][0][0], x_idx[0][-1][0], x_idx[1][0][0], x_idx[1][0][-1]

    return (start_row, stop_row), (start_col, stop_col)

def crop_image(img, rows, cols, pad):
    (start_row, stop_row) = rows
    (start_col, stop_col) = cols
    if len(img.shape)==3:
        return np.pad(img[:,start_row:stop_row, start_col:stop_col], ((0,0),(pad,pad),(pad,pad)))
    else:
        return np.pad(img[start_row:stop_row, start_col:stop_col], pad)


def raster_cell(length, width, separation, pinching=True, FL = False):
    """
    Produces a rasterised image of a cell with the intensiity of each pixel corresponding to the optical path length
    (thickness) of the cell at that point.

    :param int length: Cell length in pixels
    :param int width: Cell width in pixels
    :param int separation: An int between (0, `width`) controlling how much pinching is happening.
    :param bool pinching: Controls whether pinching is happening

    Returns
    -------

    cell : np.array
       A numpy array which contains an OPL image of the cell. Can be converted to a mask by just taking ``cell > 0``.

    """

    L = int(np.rint(length))
    W = int(np.rint(width))
    new_cell = np.zeros((L, W))
    R = (W - 1) / 2

    x_cyl = np.arange(0, 2 * R + 1, 1)
    I_cyl = np.sqrt(R ** 2 - (x_cyl - R) ** 2)
    L_cyl = L - W
    new_cell[int(W / 2):-int(W / 2), :] = I_cyl

    x_sphere = np.arange(0, int(W / 2), 1)
    sphere_Rs = np.sqrt((R) ** 2 - (x_sphere - R) ** 2)
    sphere_Rs = np.rint(sphere_Rs).astype(int)

    for c in range(len(sphere_Rs)):
        R_ = sphere_Rs[c]
        x_cyl = np.arange(0, R_, 1)
        I_cyl = np.sqrt(R_ ** 2 - (x_cyl - R_) ** 2)
        new_cell[c, int(W / 2) - sphere_Rs[c]:int(W / 2) + sphere_Rs[c]] = np.concatenate((I_cyl, I_cyl[::-1]))
        new_cell[L - c - 1, int(W / 2) - sphere_Rs[c]:int(W / 2) + sphere_Rs[c]] = np.concatenate((I_cyl, I_cyl[::-1]))

    if separation > 2 and pinching:
        S = int(np.rint(separation))
        new_cell[int((L - S) / 2) + 1:-int((L - S) / 2) - 1, :] = 0
        for c in range(int((S+1) / 2)):
            R__ = sphere_Rs[-c - 1]
            x_cyl_ = np.arange(0, R__, 1)
            I_cyl_ = np.sqrt(R__ ** 2 - (x_cyl_ - R__) ** 2)
            new_cell[int((L-S) / 2) + c + 1, int(W / 2) - R__:int(W / 2) + R__] = np.concatenate((I_cyl_, I_cyl_[::-1]))
            new_cell[-int((L-S) / 2) - c - 1, int(W / 2) - R__:int(W / 2) + R__] = np.concatenate((I_cyl_, I_cyl_[::-1]))
    new_cell = new_cell.astype(int)
    return new_cell


@njit
def OPL_to_FL(cell, density):
    """

    :param np.ndarray cell: A 2D numpy array consisting of a rasterised cell
    :param float density: Number of fluorescent molecules per volume element to sample in the cell
    :return: A cell with fluorescent reporters sampled in it
    :rtypes: np.ndarray
    """

    cell_normalised = (cell/cell.sum())
    i, j = np.arange(cell_normalised.shape[0]), np.arange(cell_normalised.shape[1])
    indices = [] #needed for njit
    for ii in i:
        for jj in j:
            indices.append((ii,jj))
    weights = cell_normalised.flatten()
    n_molecules = int(density * np.sum(cell))
    choices = np.searchsorted(np.cumsum(weights), np.random.rand(n_molecules)) # workaround for np.random.choice from
    FL_cell = np.zeros(cell.shape)
    for c in choices:
        FL_cell[indices[c]] += 1
    return FL_cell


@njit
def generate_deviation_from_CL(centreline, thickness):
    return np.arange(thickness) + centreline - int(np.ceil(thickness ))

@njit
def gen_3D_coords_from_2D(test_cells, centreline, thickness):
    return np.where(test_cells == thickness) + (generate_deviation_from_CL(centreline, thickness),)

@njit
def convert_to_3D_numba(cell):
    expanded_scene = cell
    volume_shape = expanded_scene.shape[0:] + (int(expanded_scene.max()*2),)
    test_cells = rounder(expanded_scene)
    centreline = int(expanded_scene.max() )
    cells_3D = np.zeros(volume_shape,dtype = np.ubyte)
    for t in range(int(expanded_scene.max() *2 )):
        test_coords = gen_3D_coords_from_2D(test_cells, centreline, t)
        for x, y in zip(test_coords[0], (test_coords[1])):
            for z in test_coords[2]:
                cells_3D[x, y, z] = 1
    return cells_3D

@njit
def rounder(x):
    out = np.empty_like(x)
    np.round(x, 0, out)
    return out

def convert_to_3D(cell):
    cells_3D = convert_to_3D_numba(cell)
    cells_3D = np.moveaxis(cells_3D, -1, 0)
    cells_3D[cells_3D.shape[0]//2:,:, :] = cells_3D[:cells_3D.shape[0]//2,:, :][::-1]
    return cells_3D

def draw_scene(cell_properties, do_transformation, space_size, offset, label_masks, pinching=True):
    """
    Draws a raw scene (no trench) of cells, and returns accompanying masks for training data.

    Parameters
    ----------
    cell properties : list
        A list of cell properties for that frame
    do_transformation : bool
        True if you want cells to be bent, false and cells remain straight as in the simulation
    space_size : tuple
        The xy size of the numpy array in which the space is rendered. If too small then cells will not fit. recommend using the :meth:`SyMBac.drawing.get_space_size` function to find the correct space size for your simulation
    offset : int
        A necessary parameter which offsets the drawing a number of pixels from the left hand side of the image. 30 is a good number, but if the cells are very thick, then might need increasing.
    label_masks : bool
        If true returns cell masks which are labelled (good for instance segmentation). If false returns binary masks only. I recommend leaving this as True, because you can always binarise the masks later if you want.
    pinching : bool
        Whether or not to simulate cell pinching during division

    Returns
    -------
    space, space_masks : 2D numpy array, 2D numpy array

    space : 2D numpy array
        Not to be confused with the pyglet object calledspace in some other functions. Simply a 2D numpy array with an image of cells from the input frame properties
    space_masks : 2D numy array
        The masks (labelled or bool) for that scene.

    """
    space_size = np.array(space_size)  # 1000, 200 a good value
    space = np.zeros(space_size)
    space_masks_label = np.zeros(space_size)
    space_masks_nolabel = np.zeros(space_size)
    #colour_label = [1]

    space_masks = np.zeros(space_size)
    if label_masks == False:
        space_masks = space_masks.astype(bool)

    for properties in cell_properties:
        length, width, angle, position, freq_modif, amp_modif, phase_modif, phase_mult, separation, sim_mask_label, texture_y_coordinate = properties
        position = np.array(position)
        x = np.array(position).astype(int)[0] + offset
        y = np.array(position).astype(int)[1] + offset
        OPL_cell = raster_cell(length=length, width=width, separation=separation, pinching=pinching)

        scale = 70.0
        octaves = 1
        persistence = 0.5
        lacunarity = 2
        seed = 420
        

        noise = np.zeros(OPL_cell.shape)
        for i in range(OPL_cell.shape[0]):
            for j in range(OPL_cell.shape[1]):
                noise[i][j] = pnoise2((i - OPL_cell.shape[0] + texture_y_coordinate) / 2 / scale,
                                      (j - OPL_cell.shape[1]) / 2 / scale,
                                      octaves=octaves,
                                      persistence=persistence,
                                      lacunarity=lacunarity,
                                      base=seed)

        OPL_cell = OPL_cell * (1+noise*1.5)
        
        if do_transformation:
            OPL_cell_2 = np.zeros((OPL_cell.shape[0], int(OPL_cell.shape[1] * 2)))
            midpoint = int(np.median(range(OPL_cell_2.shape[1])))
            OPL_cell_2[:,
            midpoint - int(OPL_cell.shape[1] / 2):midpoint - int(OPL_cell.shape[1] / 2) + OPL_cell.shape[1]] = OPL_cell
            roll_coords = np.array(range(OPL_cell_2.shape[0]))
            freq_mult = (OPL_cell_2.shape[0])
            amp_mult = OPL_cell_2.shape[1] / 10
            sin_transform_cell = transform_func(amp_modif, freq_modif, phase_modif)
            roll_amounts = sin_transform_cell(roll_coords, amp_mult, freq_mult, phase_mult)
            for B in roll_coords:
                OPL_cell_2[B, :] = np.roll(OPL_cell_2[B, :], roll_amounts[B])
            OPL_cell = (OPL_cell_2)


        rotated_OPL_cell = rotate(OPL_cell, -angle, resize=True, clip=False, preserve_range=True, center=(x, y))
        cell_y, cell_x = (np.array(rotated_OPL_cell.shape) / 2).astype(int)
        offset_y = rotated_OPL_cell.shape[0] - space[y - cell_y:y + cell_y, x - cell_x:x + cell_x].shape[0]
        offset_x = rotated_OPL_cell.shape[1] - space[y - cell_y:y + cell_y, x - cell_x:x + cell_x].shape[1]
        assert y > cell_y, f"Cell has {y - cell_y} negative pixels in y coordinate, try increasing your offset".format(y - cell_y)
        assert x > cell_x, f"Cell has {x - cell_x} negative pixels in x coordinate, try increasing your offset"
        space[
        y - cell_y:y + cell_y + offset_y,
        x - cell_x:x + cell_x + offset_x
        ] += rotated_OPL_cell

        def get_mask(label_masks):

            if label_masks:
                space_masks_label[y - cell_y:y + cell_y + offset_y, x - cell_x:x + cell_x + offset_x] += (rotated_OPL_cell > 0) * sim_mask_label
                #colour_label[0] += 1
                return space_masks_label
            else:
                space_masks_nolabel[y - cell_y:y + cell_y + offset_y, x - cell_x:x + cell_x + offset_x] += (
                                                                                                                   rotated_OPL_cell > 0) * 1
                return space_masks_nolabel
                # space_masks = opening(space_masks,np.ones((2,11)))

        label_mask = get_mask(True).astype(int)
        nolabel_mask = get_mask(False).astype(int)
        label_mask_fixed = np.where(nolabel_mask > 1, 0, label_mask)
        if label_masks:
            space_masks = label_mask_fixed
        else:
            mask_borders = find_boundaries(label_mask_fixed, mode="thick", connectivity=2)
            space_masks = np.where(mask_borders, 0, label_mask_fixed)
            space_masks = opening(space_masks)
            space_masks = space_masks.astype(bool)
        space = space * space_masks.astype(bool)
    return space, space_masks


def get_distance(vertex1, vertex2):
    """
    Get euclidian distance between two sets of vertices.


    :param tuple(float, float) vertex1: Vertex 1
    :param tuple(float, float) vertex2: Vertex 2

    :return: Absolute distance between two points
    :rtype: float
    """
    return abs(np.sqrt((vertex1[0] - vertex2[0]) ** 2 + (vertex1[1] - vertex2[1]) ** 2))


def find_farthest_vertices(vertex_list):
    """Given a list of vertices, find the pair of vertices which are farthest from each other

    Parameters
    ----------
    vertex_list : list(tuple(float, float))
        List of pairs of vertices [(x,y), (x,y), ...]

    Returns
    -------
    array(tuple(float, float))
        The two vertices maximally far apart
    """
    vertex_combs = list(itertools.combinations(vertex_list, 2))
    distance = 0
    farthest_vertices = 0
    for vertex_comb in vertex_combs:
        distance_ = get_distance(vertex_comb[0], vertex_comb[1])
        if distance_ > distance:
            distance = distance_
            farthest_vertices = vertex_comb
    return np.array(farthest_vertices)


def get_midpoint(vertex1, vertex2):
    """

    Get the midpoint between two vertices.

    :param tuple(float, float) vertex1: Vertex 1
    :param tuple(float, float) vertex2: Vertex 2
    :return: Midpoint between vertex 1 and 2
    :rtype: tuple(float, float)
    """
    x_mid = (vertex1[0] + vertex2[0]) / 2
    y_mid = (vertex1[1] + vertex2[1]) / 2
    return np.array([x_mid, y_mid])


def vertices_slope(vertex1, vertex2):
    """
    Get the slope between two vertices

    :param tuple(float, float) vertex1: Vertex 1
    :param tuple(float, float) vertex2: Vertex 2
    :return: Slope between vertex 1 and 2
    :rtype: float
    """
    return (vertex1[1] - vertex2[1]) / (vertex1[0] - vertex2[0])


def midpoint_intercept(vertex1, vertex2):
    """
    Get the y-intercept of the line connecting two vertices

    :param tuple(float, float) vertex1: Vertex 1
    :param tuple(float, float) vertex2: Vertex 2
    :return: Y indercept of line between vertex 1 and 2
    :rtype: float
    """
    midpoint = get_midpoint(vertex1, vertex2)
    slope = vertices_slope(vertex1, vertex2)
    intercept = midpoint[1] - (slope * midpoint[0])
    return intercept


def get_centroid(vertices):
    """Return the centroid of a list of vertices 
    
    :param list(tuple(float, float)) vertices: List of tuple of vertices where each tuple is (x, y)
    :return: Centroid of the vertices.
    :rtype: tuple(float, float)
    """
    return np.sum(vertices, axis=0) / len(vertices)


def place_cell(length, width, angle, position, space):
    """Creates a cell and places it in the pymunk space

    Parameters
    ----------
    length : float
        length of the cell
    width : float
        width of the cell
    angle : float
        rotation of the cell in radians counterclockwise
    position : tuple
        x,y coordinates of the cell centroid
    space : pymunk.space.Space
        Pymunk space to place the cell in

    Returns
    -------
    nothing, updates space

    """
    angle = np.rad2deg(angle)
    x, y = np.array(position).astype(int)
    OPL_cell = raster_cell(length=length, width=width)
    rotated_OPL_cell = rotate(OPL_cell, angle, resize=True, clip=False, preserve_range=True)
    cell_y, cell_x = (np.array(rotated_OPL_cell.shape) / 2).astype(int)
    offset_y = rotated_OPL_cell.shape[0] - space[y - cell_y:y + cell_y, x - cell_x:x + cell_x].shape[0]
    offset_x = rotated_OPL_cell.shape[1] - space[y - cell_y:y + cell_y, x - cell_x:x + cell_x].shape[1]
    space[y - cell_y + 100:y + cell_y + offset_y + 100,
    x - cell_x + 100:x + cell_x + offset_x + 100] += rotated_OPL_cell


def transform_func(amp_modif, freq_modif, phase_modif):
    def perm_transform_func(x, amp_mult, freq_mult, phase_mult):
        return (amp_mult * amp_modif * np.cos(
            (x / (freq_mult * freq_modif) - phase_mult * phase_modif) * np.pi)).astype(int)

    return perm_transform_func


def scene_plotter(scene_array, output_dir, name, a, matplotlib_draw):
    if matplotlib_draw == True:
        plt.figure(figsize=(3, 10))
        plt.imshow(scene_array)
        plt.tight_layout()
        plt.savefig(output_dir + "/{}_{}.png".format(name, str(a).zfill(3)))
        plt.clf()
        plt.close('all')
    else:
        im = Image.fromarray(scene_array.astype(np.uint8))
        im.save(output_dir + "/{}_{}.tif".format(name, str(a).zfill(3)))


def make_images_same_shape(real_image, synthetic_image, rescale_int=True):
    """ Makes a synthetic image the same shape as the real image """

    assert real_image.shape[0] < synthetic_image.shape[
        0], "Real image has a higher diemsion on axis 0, increase y_border_expansion_coefficient"
    assert real_image.shape[1] < synthetic_image.shape[
        1], "Real image has a higher diemsion on axis 1, increase x_border_expansion_coefficient"

    x_diff = synthetic_image.shape[1] - real_image.shape[1]
    remove_from_left, remove_from_right = div_odd(x_diff)
    y_diff = synthetic_image.shape[0] - real_image.shape[0]
    if real_image.shape[1] % 2 == 0:
        if synthetic_image.shape[1] % 2 == 0:
            if y_diff > 0:
                synthetic_image = synthetic_image[y_diff:, remove_from_left - 1:-remove_from_right]
            else:
                synthetic_image = synthetic_image[:, remove_from_left:-remove_from_right]
                real_image = real_image[abs(y_diff):, :]
        elif synthetic_image.shape[1] % 2 == 1:
            if y_diff > 0:
                synthetic_image = synthetic_image[y_diff:, remove_from_left:-remove_from_right]
            else:
                synthetic_image = synthetic_image[:, remove_from_left:-remove_from_right]
                real_image = real_image[abs(y_diff):, :]
    elif real_image.shape[1] % 2 == 1:
        if synthetic_image.shape[1] % 2 == 0:
            if y_diff > 0:
                synthetic_image = synthetic_image[y_diff:, remove_from_left:-remove_from_right]
            else:
                synthetic_image = synthetic_image[:, remove_from_left:-remove_from_right]
                real_image = real_image[abs(y_diff):, :]
        elif synthetic_image.shape[1] % 2 == 1:
            if y_diff > 0:
                synthetic_image = synthetic_image[y_diff:, remove_from_left - 1:-remove_from_right]
            else:
                synthetic_image = synthetic_image[:, remove_from_left:-remove_from_right]
                real_image = real_image[abs(y_diff):, :]

    if rescale_int:
        real_image = rescale_intensity(real_image.astype(np.float32), out_range=(0, 1))
        synthetic_image = rescale_intensity(synthetic_image.astype(np.float32), out_range=(0, 1))
    return real_image, synthetic_image


def get_space_size(cell_timeseries_properties):
    """
    :param cell_timeseries_properties: A list of cell properties over time. Generated from :meth:`SyMBac.simulation.Simulation.draw_simulation_OPL`
    :return: Iterates through the simulation timeseries properties, finds the extreme cell positions and retrieves the required image size to fit all cells into.
    :rtype: tuple(float, float)
    """
    max_x, max_y = 0, 0
    for timepoint in cell_timeseries_properties:
        for cell in timepoint:
            x_, y_ = np.ceil(cell[3]).astype(int)
            length_ = np.ceil(cell[0]).astype(int)
            width_ = np.ceil(cell[1]).astype(int)
            max_y_ = y_ + length_
            max_x_ = x_ + width_
            if max_x_ > max_x:
                max_x = max_x_
            if max_y_ > max_y:
                max_y = max_y_
    return (int(1.2 * max_y), int(1.5 * max_x))


def clean_up_mask(mask):
    return remove_small_objects(label(mask))



def draw_simulation_OPL(simulation, do_transformation = True, label_masks = True):


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
    simulation.main_segments = get_trench_segments(simulation.space)
    ID_props = generate_curve_props(simulation.cell_timeseries)

    cell_timeseries_properties = Parallel(n_jobs=-1)(
        delayed(gen_cell_props_for_draw)(a, ID_props) for a in tqdm(simulation.cell_timeseries, desc='Extracting cell properties from the simulation'))

    space_size = get_space_size(cell_timeseries_properties)
    offset = 30
    scenes = Parallel(n_jobs=-1)(delayed(draw_scene)(
    cell_properties, do_transformation, space_size, offset, label_masks) for cell_properties in tqdm(
        cell_timeseries_properties, desc='Rendering cell optical path lengths'))
    OPL_scenes = [_[0] for _ in scenes]
    masks = [_[1] for _ in scenes]

    return OPL_scenes, masks