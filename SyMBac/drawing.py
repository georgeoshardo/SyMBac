import importlib
import itertools
import warnings
import napari
import numpy as np
from ipywidgets import interactive, fixed
import psfmodels as psfm
from matplotlib import pyplot as plt
from skimage.transform import rescale, rotate
from skimage.util import random_noise
from joblib import Parallel, delayed
from skimage.morphology import opening
from tqdm import tqdm
from skimage import draw
from skimage.exposure import match_histograms, rescale_intensity
from SyMBac.PSF import get_fluorescence_kernel, get_phase_contrast_kernel
import os
import skimage
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from numpy import fft
from PIL import Image
import copy
from SyMBac.PSF import PSF_generator

div_odd = lambda n: (n // 2, n // 2 + 1)
perc_diff = lambda a, b: (a - b) / b * 100

if importlib.util.find_spec("cupy") is None:
    from scipy.signal import convolve2d as cuconvolve

    warnings.warn("Could not load CuPy for SyMBac, are you using a GPU? Defaulting to CPU convolution.")


    def convolve_rescale(image, kernel, rescale_factor, rescale_int):
        """
        Convolves an image with a kernel, and rescales it to the correct size. 

        Parameters
        ----------
        image : numpy.ndarray
            The image
        kernel : 2D numpy array
            The kernel
        rescale_factor : int
            Typicall 1/resize_amount. So 1/3 will scale the image down by a factor of 3. We do this because we render the image and kernel at high resolution, so that we can do the convolution at high resolution.
        rescale_int : bool
            If True, rescale the intensities between 0 and 1 and return a float32 numpy array of the convolved downscaled image.

        Returns
        -------
        outupt : 2D numpy array
            The output of the convolution rescale operation
        """

        output = cuconvolve(image, kernel, mode="same")
        # output = output.get()
        output = rescale(output, rescale_factor, anti_aliasing=False)

        if rescale_int:
            output = rescale_intensity(output.astype(np.float32), out_range=(0, 1))
        return output
else:
    import cupy as cp
    from cupyx.scipy.ndimage import convolve as cuconvolve


    def convolve_rescale(image, kernel, rescale_factor, rescale_int):
        """
        Convolves an image with a kernel, and rescales it to the correct size. 

        Parameters
        ----------
        image : 2D numpy array
            The image
        kernel : 2D numpy array
            The kernel
        rescale_factor : int
            Typicall 1/resize_amount. So 1/3 will scale the image down by a factor of 3. We do this because we render the image and kernel at high resolution, so that we can do the convolution at high resolution.
        rescale_int : bool
            If True, rescale the intensities between 0 and 1 and return a float32 numpy array of the convolved downscaled image.

        Returns
        -------
        outupt : 2D numpy array
            The output of the convolution rescale operation
        """

        output = cuconvolve(cp.array(image), cp.array(kernel))
        output = output.get()
        output = rescale(output, rescale_factor, anti_aliasing=False)

        if rescale_int:
            output = rescale_intensity(output.astype(np.float32), out_range=(0, 1))
        return output


def generate_curve_props(cell_timeseries):
    """
    Generates individual cell curvature properties. 3 parameters for each cell, which are passed to a cosine function to modulate the cell's curvature. 
    
    Parameters
    ---------
    cell_timeseries : list(cell_properties)
        The output of run_simulation()
    
    Returns
    -------
    outupt : A numpy array of unique curvature properties for each cell in the simulation
    """

    # Get unique cell IDs
    IDs = []
    for cell_list in cell_timeseries:
        for cell in cell_list:
            IDs.append(cell.ID)
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
        width = cell.width
        separation = cell.pinching_sep
        # angle = np.arctan(vertices_slope(farthest_vertices[0], farthest_vertices[1]))
        angle = np.arctan2((farthest_vertices[0] - farthest_vertices[1])[1],
                           (farthest_vertices[0] - farthest_vertices[1])[0])
        angle = np.rad2deg(angle) + 90

        ID, freq_modif, amp_modif, phase_modif = ID_props[ID_props[:, 0] == cell.ID][0]
        phase_mult = 20
        cell_properties.append([length, width, angle, centroid, freq_modif, amp_modif, phase_modif, phase_mult,
                                separation])
    return cell_properties


def raster_cell(length, width, separation, pinching=True):
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

    if separation > 1 and pinching:
        S = int(np.rint(separation))
        new_cell[int((L-S) / 2):-int((L-S) / 2), :] = 0
        for c in range(int(S/2)):
            R__ = sphere_Rs[-c-1]
            x_cyl_ = np.arange(0, R__, 1)
            I_cyl_ = np.sqrt(R__ ** 2 - (x_cyl_ - R__) ** 2)
            new_cell[int((L-S) / 2) + c, int(W / 2) - R__:int(W / 2) + R__] = np.concatenate((I_cyl_, I_cyl_[::-1]))
            new_cell[int((L+S) / 2) - c - 1, int(W / 2) - R__:int(W / 2) + R__] = np.concatenate(
                (I_cyl_, I_cyl_[::-1]))
    new_cell = new_cell.astype(int)
    return new_cell


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
        The xy size of the numpy array in which the space is rendered. If too small then cells will not fit. recommend using the get_space_size() function to find the correct space size for your simulation
    offset : int
        A necessary parameter which offsets the drawing a number of pixels from the left hand side of the image. 30 is a good number, but if the cells are very thick, then might need increasing.
    label_masks : bool
        If true returns cell masks which are labelled (good for instance segmentation). If false returns binary masks only. I recommend leaving this as True, because you can always binarise the masks later if you want.

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
    colour_label = [1]

    space_masks = np.zeros(space_size)
    if label_masks == False:
        space_masks = space_masks.astype(bool)

    for properties in cell_properties:
        length, width, angle, position, freq_modif, amp_modif, phase_modif, phase_mult, separation = properties
        position = np.array(position)
        x = np.array(position).astype(int)[0] + offset
        y = np.array(position).astype(int)[1] + offset
        OPL_cell = raster_cell(length=length, width=width, separation=separation, pinching=pinching)

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
        assert y > cell_y, "Cell has {} negative pixels in y coordinate, try increasing your offset".format(y - cell_y)
        assert x > cell_x, "Cell has negative pixels in x coordinate, try increasing your offset"
        space[
        y - cell_y:y + cell_y + offset_y,
        x - cell_x:x + cell_x + offset_x
        ] += rotated_OPL_cell

        def get_mask(label_masks):

            if label_masks:
                space_masks_label[y - cell_y:y + cell_y + offset_y, x - cell_x:x + cell_x + offset_x] += (
                                                                                                                 rotated_OPL_cell > 0) * \
                                                                                                         colour_label[0]
                colour_label[0] += 1
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

    Parameters
    ----------
    vertex1 : 2-tuple
        x,y coordinates of a vertex
    vertex2 : 2-tuple
        x,y coordinates of a vertex

    Returns
    -------
    float : absolute distance between two points
    """
    return abs(np.sqrt((vertex1[0] - vertex2[0]) ** 2 + (vertex1[1] - vertex2[1]) ** 2))


def find_farthest_vertices(vertex_list):
    """Given a list of vertices, find the pair of vertices which are farthest from each other

    Parameters
    ----------
    vertex_list : list(2-tuple, 2-tuple ... )
        List of pairs of vertices [(x,y), (x,y), ...]

    Returns
    -------
    array(2-tuple, 2-tuple)
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
    Get the midpoint between two vertices
    """
    x_mid = (vertex1[0] + vertex2[0]) / 2
    y_mid = (vertex1[1] + vertex2[1]) / 2
    return np.array([x_mid, y_mid])


def vertices_slope(vertex1, vertex2):
    """
    Get the slope between two vertices
    """
    return (vertex1[1] - vertex2[1]) / (vertex1[0] - vertex2[0])


def midpoint_intercept(vertex1, vertex2):
    """
    Get the y-intercept of the line connecting two vertices
    """
    midpoint = get_midpoint(vertex1, vertex2)
    slope = vertices_slope(vertex1, vertex2)
    intercept = midpoint[1] - (slope * midpoint[0])
    return intercept


def get_centroid(vertices):
    """Return the centroid of a list of vertices 
    
    Keyword arguments:
    vertices -- A list of tuples containing x,y coordinates.

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
    """ Makes a synthetic image the same shape as the real image"""

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
    """Iterates through the simulation timeseries properties, 
    finds the extreme cell positions and retrieves the required 
    image size to fit all cells into"""
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


class Renderer:
    """

            y_border_expansioon_coefficient : int
            Another offset-like argument. Multiplies the size of the image on each side by this value. 3 is a good starting
            value because you want the image to be relatively larger than the PSF which you are convolving over it.
        x_border_expansioon_coefficient : int
            Another offset-like argument. Multiplies the size of the image on each side by this value. 3 is a good starting
            value because you want the image to be relatively larger than the PSF which you are convolving over it.

    """


    def __init__(self, simulation, PSF, real_image, camera=None):
        self.real_image = real_image
        self.PSF = PSF
        self.simulation = simulation
        self.real_image = real_image
        self.camera=camera
        media_multiplier = 30
        cell_multiplier = 1
        device_multiplier = -50
        self.y_border_expansion_coefficient = 2
        self.x_border_expansion_coefficient = 2

        temp_expanded_scene, temp_expanded_scene_no_cells, temp_expanded_mask = self.generate_PC_OPL(
            scene=simulation.OPL_scenes[-1],
            mask=simulation.masks[-1],
            media_multiplier=media_multiplier,
            cell_multiplier=cell_multiplier,
            device_multiplier=cell_multiplier,
            y_border_expansion_coefficient=self.y_border_expansion_coefficient,
            x_border_expansion_coefficient=self.x_border_expansion_coefficient,
            defocus=30
        )

        ### Generate temporary image to make same shape
        self.PSF.calculate_PSF()
        if "3d" in self.PSF.mode.lower():
            temp_kernel = self.PSF.kernel.mean(axis=0)
        else:
            temp_kernel = self.PSF.kernel

        convolved = convolve_rescale(temp_expanded_scene, temp_kernel, 1 / simulation.resize_amount, rescale_int=True)
        self.real_resize, self.expanded_resized = make_images_same_shape(real_image, convolved, rescale_int=True)
        mean_error = []
        media_error = []
        cell_error = []
        device_error = []

        mean_var_error = []
        media_var_error = []
        cell_var_error = []
        device_var_error = []

        self.error_params = (mean_error, media_error, cell_error, device_error, mean_var_error, media_var_error, cell_var_error, device_var_error)

    def select_intensity_napari(self):
        viewer = napari.view_image(self.real_resize)
        self.media_label = viewer.add_labels(np.zeros(self.real_resize.shape).astype(int), name="Media")
        self.cell_label = viewer.add_labels(np.zeros(self.real_resize.shape).astype(int), name="Cell")
        self.device_label = viewer.add_labels(np.zeros(self.real_resize.shape).astype(int), name="Device")


    def generate_test_comparison(self, media_multiplier=75, cell_multiplier=1.7, device_multiplier=29, sigma=8.85,
                                 scene_no=-1, match_fourier=False, match_histogram=True, match_noise=False,
                                 debug_plot=False, noise_var=0.001,
                                 resize_amount=None, real_image=None, defocus=3.0):
        """

        Takes all the parameters we've defined and calculated, and uses them to finally generate a synthetic image.

        Parameters
        ----------
        media_multiplier : float
            Intensity multiplier for media (the area between cells which isn't the device)
        cell_multiplier : float
            Intensity multiplier for cell
        device_multiplier : float
            Intensity multiplier for device
        sigma : float
            Radius of a gaussian which simulates PSF apodisation
        scene_no : int in range(len(cell_timeseries_properties))
            The index of which scene to render
        scale : float
            The micron/pixel value of the image
        match_fourier : bool
            If true, use sfmatch to match the rotational fourier spectrum of the synthetic image to a real image sample
        match_histogram : bool
            If true, match the intensity histogram of a synthetic image to a real image
        offset : int
            The same offset value from draw_scene
        debug_plot : bool
            True if you want to see a quick preview of the rendered synthetic image
        noise_var : float
            The variance for the simulated camera noise (gaussian)
        kernel : SyMBac.PSF.PSF_generator
            A kernel object from SyMBac.PSF.PSF_generator
        resize_amount : int
            The upscaling factor to render the image by. E.g a resize_amount of 3 will interally render the image at 3x
            resolution before convolving and then downsampling the image. Values >2 are recommended.
        real_image : 2D numpy array
            A sample real image from the experiment you are trying to replicate
        image_params : tuple
            A tuple of parameters which describe the intensities and variances of the real image, in this order:
            (real_media_mean, real_cell_mean, real_device_mean, real_means, real_media_var, real_cell_var, real_device_var,
                real_vars).
        error_params : tuple
            A tuple of parameters which characterises the error between the intensities in the real image and the synthetic
            image, in this order: (mean_error,media_error,cell_error,device_error,mean_var_error,media_var_error,
            cell_var_error,device_var_error). I have given an example of their calculation in the example notebooks.

        fluorescence : bool
            If true converts image to a fluorescence (hides the trench and swaps to the fluorescence PSF).
        defocus : float
            Simulated optical defocus by convolving the kernel with a 2D gaussian of radius defocus.

        Returns
        -------
        noisy_img : 2D numpy array
            The final simulated microscope image
        expanded_mask_resized_reshaped : 2D numpy array
            The final image's accompanying masks

        """

        expanded_scene, expanded_scene_no_cells, expanded_mask = self.generate_PC_OPL(
            scene=self.simulation.OPL_scenes[scene_no],
            mask=self.simulation.masks[scene_no],
            media_multiplier=media_multiplier,
            cell_multiplier=cell_multiplier,
            device_multiplier=device_multiplier,
            x_border_expansion_coefficient=self.x_border_expansion_coefficient,
            y_border_expansion_coefficient=self.y_border_expansion_coefficient,
            defocus=defocus
        )

        R, W, radius, scale, NA, n, _, λ = self.PSF.R, self.PSF.W, self.PSF.radius, self.PSF.scale, self.PSF.NA, self.PSF.n, self.PSF.apo_sigma, self.PSF.wavelength

        real_media_mean, real_cell_mean, real_device_mean, real_means, real_media_var, real_cell_var, real_device_var, real_vars = self.image_params
        mean_error, media_error, cell_error, device_error, mean_var_error, media_var_error, cell_var_error, device_var_error = self.error_params

        if self.PSF.mode == "3d fluo":  # Full 3D PSF model
            def generate_deviation_from_CL(centreline, thickness):
                return np.arange(thickness) + centreline - int(np.ceil(thickness / 2))

            def gen_3D_coords_from_2D(centreline, thickness):
                return np.where(test_cells == thickness) + (generate_deviation_from_CL(centreline, thickness),)

            volume_shape = expanded_scene.shape[0:] + (int(expanded_scene.max()),)
            test_cells = np.round(expanded_scene)
            centreline = int(expanded_scene.max() / 2)
            cells_3D = np.zeros(volume_shape)
            for t in range(int(expanded_scene.max() * 2)):
                test_coords = gen_3D_coords_from_2D(centreline, t)
                for x, y in zip(test_coords[0], (test_coords[1])):
                    for z in test_coords[2]:
                        cells_3D[x, y, z] = 1
            cells_3D = np.moveaxis(cells_3D, -1, 0)
            psf = psfm.make_psf(volume_shape[2], radius * 2, dxy=scale, dz=scale, pz=0, ni=n, wvl=λ, NA=NA)
            convolved = np.zeros(cells_3D.shape)
            for x in range(len(cells_3D)):
                temp_conv = cuconvolve(cp.array(cells_3D[x]), cp.array(psf[x])).get()
                convolved[x] = temp_conv
            convolved = convolved.sum(axis=0)
            convolved = rescale(convolved, 1 / resize_amount, anti_aliasing=False)
            convolved = rescale_intensity(convolved.astype(np.float32), out_range=(0, 1))
        else:
            kernel = self.PSF.kernel
            if defocus > 0:
                kernel = gaussian_filter(kernel, defocus, mode="reflect")
            convolved = convolve_rescale(expanded_scene, kernel, 1 / resize_amount, rescale_int=True)

        real_resize, expanded_resized = make_images_same_shape(real_image, convolved, rescale_int=True)
        fftim1 = fft.fftshift(fft.fft2(real_resize))
        angs, mags = cart2pol(np.real(fftim1), np.imag(fftim1))

        if match_fourier and not match_histogram:
            matched = sfMatch([real_resize, expanded_resized], tarmag=mags)[1]
            matched = lumMatch([real_resize, matched], None, [np.mean(real_resize), np.std(real_resize)])[1]
        else:
            matched = expanded_resized

        if match_histogram and match_fourier:
            matched = sfMatch([real_resize, matched], tarmag=mags)[1]
            matched = lumMatch([real_resize, matched], None, [np.mean(real_resize), np.std(real_resize)])[1]
            matched = match_histograms(matched, real_resize, multichannel=False)
        else:
            pass
        if match_histogram:
            matched = match_histograms(matched, real_resize, multichannel=False)
        else:
            pass

        if self.camera:  # Camera noise simulation
            baseline, sensitivity, dark_noise = self.camera.baseline, self.camera.sensitivity, self.camera.dark_noise
            rng = np.random.default_rng(2)
            matched = matched / (matched.max() / real_image.max()) / sensitivity
            matched = rng.poisson(matched)
            noisy_img = matched + rng.normal(loc=baseline, scale=dark_noise, size=matched.shape)
        else:  # Ad hoc noise mathcing
            noisy_img = random_noise(rescale_intensity(matched), mode="poisson")
            noisy_img = random_noise(rescale_intensity(noisy_img), mode="gaussian", mean=0, var=noise_var, clip=False)

        if match_noise:
            noisy_img = match_histograms(noisy_img, real_resize, multichannel=False)
        else:
            pass
        noisy_img = rescale_intensity(noisy_img.astype(np.float32), out_range=(0, 1))

        ## getting the cell mask to the right shape
        expanded_mask_resized = rescale(expanded_mask, 1 / resize_amount, anti_aliasing=False, preserve_range=True,
                                        order=0)
        if len(np.unique(expanded_mask_resized)) > 2:
            _, expanded_mask_resized_reshaped = make_images_same_shape(real_image, expanded_mask_resized,
                                                                       rescale_int=False)
        else:
            _, expanded_mask_resized_reshaped = make_images_same_shape(real_image, expanded_mask_resized,
                                                                       rescale_int=True)

        expanded_media_mask = rescale(
            (expanded_scene_no_cells == device_multiplier) ^ (expanded_scene - expanded_scene_no_cells).astype(bool),
            1 / resize_amount, anti_aliasing=False)
        real_resize, expanded_media_mask = make_images_same_shape(real_image, expanded_media_mask, rescale_int=True)
        just_media = expanded_media_mask * noisy_img

        expanded_cell_pseudo_mask = (expanded_scene - expanded_scene_no_cells).astype(bool)
        expanded_cell_pseudo_mask = rescale(expanded_cell_pseudo_mask, 1 / resize_amount, anti_aliasing=False)

        real_resize, expanded_cell_pseudo_mask = make_images_same_shape(real_image, expanded_cell_pseudo_mask,
                                                                        rescale_int=True)
        just_cells = expanded_cell_pseudo_mask * noisy_img

        expanded_device_mask = expanded_scene_no_cells

        expanded_device_mask = rescale(expanded_device_mask, 1 / resize_amount, anti_aliasing=False)
        real_resize, expanded_device_mask = make_images_same_shape(real_image, expanded_device_mask, rescale_int=True)
        just_device = expanded_device_mask * noisy_img

        simulated_means = np.array([just_media[np.where(just_media)].mean(), just_cells[np.where(just_cells)].mean(),
                                    just_device[np.where(just_device)].mean()])
        simulated_vars = np.array([just_media[np.where(just_media)].var(), just_cells[np.where(just_cells)].var(),
                                   just_device[np.where(just_device)].var()])
        mean_error.append(perc_diff(np.mean(noisy_img), np.mean(real_resize)))
        mean_var_error.append(perc_diff(np.var(noisy_img), np.var(real_resize)))
        if "fluo" in self.PSF.mode.lower():
            pass
        else:
            media_error.append(perc_diff(simulated_means[0], real_media_mean))
            cell_error.append(perc_diff(simulated_means[1], real_cell_mean))
            device_error.append(perc_diff(simulated_means[2], real_device_mean))

            media_var_error.append(perc_diff(simulated_vars[0], real_media_var))
            cell_var_error.append(perc_diff(simulated_vars[1], real_cell_var))
            device_var_error.append(perc_diff(simulated_vars[2], real_device_var))
        if debug_plot:
            fig = plt.figure(figsize=(15, 5))
            ax1 = plt.subplot2grid((1, 8), (0, 0), colspan=1, rowspan=1)
            ax2 = plt.subplot2grid((1, 8), (0, 1), colspan=1, rowspan=1)
            ax3 = plt.subplot2grid((1, 8), (0, 2), colspan=3, rowspan=1)
            ax4 = plt.subplot2grid((1, 8), (0, 5), colspan=3, rowspan=1)
            ax1.imshow(noisy_img, cmap="Greys_r")
            ax1.set_title("Synthetic")
            ax1.axis("off")
            ax2.imshow(real_resize, cmap="Greys_r")
            ax2.set_title("Real")
            ax2.axis("off")
            ax3.plot(mean_error)
            ax3.plot(media_error)
            ax3.plot(cell_error)
            ax3.plot(device_error)
            ax3.legend(["Mean error", "Media error", "Cell error", "Device error"])
            ax3.set_title("Intensity Error")
            ax3.hlines(0, ax3.get_xlim()[0], ax3.get_xlim()[1], color="k", linestyles="dotted")
            ax4.plot(mean_var_error)
            ax4.plot(media_var_error)
            ax4.plot(cell_var_error)
            ax4.plot(device_var_error)
            ax4.legend(["Mean error", "Media error", "Cell error", "Device error"])
            ax4.set_title("Variance Error")
            ax4.hlines(0, ax4.get_xlim()[0], ax4.get_xlim()[1], color="k", linestyles="dotted")
            fig.tight_layout()
            plt.show()
            plt.close()
        else:
            return noisy_img, expanded_mask_resized_reshaped.astype(int)

    def generate_PC_OPL(self, scene, mask, media_multiplier, cell_multiplier, device_multiplier,
                        y_border_expansion_coefficient, x_border_expansion_coefficient, defocus):
        """
        Takes a scene drawing, adds the trenches and colours all parts of the image to generate a first-order phase contrast
        image, uncorrupted (unconvolved) by the phase contrat optics. Also has a fluorescence parameter to quickly switch to
        fluorescence if you want.

        Parameters
        ----------
        main_segments : list
            A list of the trench segments, used for drawing the trench
        offset : int
            The same offset from the draw_scene function. Used to know the cell offset.
        scene : 2D numpy array
            A scene image
        mask : 2D numpy array
            The mask for the scene
        media_multiplier : float
            Intensity multiplier for media (the area between cells which isn't the device)
        cell_multiplier : float
            Intensity multiplier for cell
        device_multiplier : float
            Intensity multiplier for device
        y_border_expansion_coefficient : int
            Another offset-like argument. Multiplies the size of the image on each side by this value. 3 is a good starting
            value because you want the image to be relatively larger than the PSF which you are convolving over it.
        x_border_expansion_coefficient : int
            Another offset-like argument. Multiplies the size of the image on each side by this value. 3 is a good starting
            value because you want the image to be relatively larger than the PSF which you are convolving over it.
        fluorescence : bool
            If true converts image to a fluorescence (hides the trench and swaps to the fluorescence PSF).
        defocus : float
            Simulated optical defocus by convolving the kernel with a 2D gaussian of radius defocus.

        Returns
        -------
        expanded_scene : 2D numpy array
            A large (expanded on x and y axis) image of cells in a trench, but unconvolved. (The raw PC image before
            convolution)
        expanded_scene_no_cells : 2D numpy array
            Same as expanded_scene, except with the cells removed (this is necessary for later intensity tuning)
        expanded_mask : 2D numpy array
            The masks for the expanded scene
        """

        def get_OPL_image(scene, mask, media_multiplier, cell_multiplier, device_multiplier,
                          y_border_expansion_coefficient, x_border_expansion_coefficient, defocus):
            print(self.simulation)
            segment_1_top_left = (0 + self.simulation.offset, int(self.simulation.main_segments.iloc[0]["bb"][0] + self.simulation.offset))
            segment_1_bottom_right = (
                int(self.simulation.main_segments.iloc[0]["bb"][3] + self.simulation.offset), int(self.simulation.main_segments.iloc[0]["bb"][2] + self.simulation.offset))

            segment_2_top_left = (0 + self.simulation.offset, int(self.simulation.main_segments.iloc[1]["bb"][0] + self.simulation.offset))
            segment_2_bottom_right = (
                int(self.simulation.main_segments.iloc[1]["bb"][3] + self.simulation.offset), int(self.simulation.main_segments.iloc[1]["bb"][2] + self.simulation.offset))

            if "fluo" in self.PSF.mode.lower():
                test_scene = np.zeros(scene.shape)
                media_multiplier = -1 * device_multiplier
            else:
                test_scene = np.zeros(scene.shape) + device_multiplier
                rr, cc = draw.rectangle(start=segment_1_top_left, end=segment_1_bottom_right, shape=test_scene.shape)
                test_scene[rr, cc] = 1 * media_multiplier
                rr, cc = draw.rectangle(start=segment_2_top_left, end=segment_2_bottom_right, shape=test_scene.shape)
                test_scene[rr, cc] = 1 * media_multiplier
                circ_midpoint_y = (segment_1_top_left[1] + segment_2_bottom_right[1]) / 2
                radius = (segment_1_top_left[1] - self.simulation.offset - (segment_2_bottom_right[1] - self.simulation.offset)) / 2
                circ_midpoint_x = (self.simulation.offset) + radius

                rr, cc = draw.rectangle(start=segment_2_top_left, end=(circ_midpoint_x, segment_1_top_left[1]),
                                        shape=test_scene.shape)
                test_scene[rr.astype(int), cc.astype(int)] = 1 * media_multiplier
                rr, cc = draw.disk(center=(circ_midpoint_x, circ_midpoint_y), radius=radius, shape=test_scene.shape)
                rr_semi = rr[rr < (circ_midpoint_x + 1)]
                cc_semi = cc[rr < (circ_midpoint_x + 1)]
                test_scene[rr_semi, cc_semi] = device_multiplier
            no_cells = copy.deepcopy(test_scene)

            test_scene += scene * cell_multiplier
            if "fluo" in self.PSF.mode.lower():
                pass
            else:
                test_scene = np.where(no_cells != media_multiplier, test_scene, media_multiplier)
            test_scene = test_scene[segment_2_top_left[0]:segment_1_bottom_right[0],
                         segment_2_top_left[1]:segment_1_bottom_right[1]]

            mask = np.where(no_cells != media_multiplier, mask, 0)
            mask_resized = mask[segment_2_top_left[0]:segment_1_bottom_right[0],
                           segment_2_top_left[1]:segment_1_bottom_right[1]]

            no_cells = no_cells[segment_2_top_left[0]:segment_1_bottom_right[0],
                       segment_2_top_left[1]:segment_1_bottom_right[1]]
            expanded_scene_no_cells = np.zeros((int(no_cells.shape[0] * y_border_expansion_coefficient),
                                                int(no_cells.shape[
                                                        1] * x_border_expansion_coefficient))) + media_multiplier
            expanded_scene_no_cells[expanded_scene_no_cells.shape[0] - no_cells.shape[0]:,
            int(expanded_scene_no_cells.shape[1] / 2 - int(test_scene.shape[1] / 2)):int(
                expanded_scene_no_cells.shape[1] / 2 - int(test_scene.shape[1] / 2)) + no_cells.shape[1]] = no_cells
            if "fluo" in self.PSF.mode.lower():
                expanded_scene = np.zeros((int(test_scene.shape[0] * y_border_expansion_coefficient),
                                           int(test_scene.shape[1] * x_border_expansion_coefficient)))
                expanded_scene[expanded_scene.shape[0] - test_scene.shape[0]:,
                int(expanded_scene.shape[1] / 2 - int(test_scene.shape[1] / 2)):int(
                    expanded_scene.shape[1] / 2 - int(test_scene.shape[1] / 2)) + test_scene.shape[1]] = test_scene
            else:
                expanded_scene = np.zeros((int(test_scene.shape[0] * y_border_expansion_coefficient),
                                           int(test_scene.shape[
                                                   1] * x_border_expansion_coefficient))) + media_multiplier
                expanded_scene[expanded_scene.shape[0] - test_scene.shape[0]:,
                int(expanded_scene.shape[1] / 2 - int(test_scene.shape[1] / 2)):int(
                    expanded_scene.shape[1] / 2 - int(test_scene.shape[1] / 2)) + test_scene.shape[1]] = test_scene

            expanded_mask = np.zeros((int(test_scene.shape[0] * y_border_expansion_coefficient),
                                      int(test_scene.shape[1] * x_border_expansion_coefficient)))
            expanded_mask[expanded_mask.shape[0] - test_scene.shape[0]:,
            int(expanded_mask.shape[1] / 2 - int(test_scene.shape[1] / 2)):int(
                expanded_mask.shape[1] / 2 - int(test_scene.shape[1] / 2)) + test_scene.shape[1]] = mask_resized

            return expanded_scene, expanded_scene_no_cells, expanded_mask

        expanded_scene, expanded_scene_no_cells, expanded_mask = get_OPL_image(scene, mask,
                                                                               media_multiplier, cell_multiplier,
                                                                               device_multiplier,
                                                                               y_border_expansion_coefficient,
                                                                               x_border_expansion_coefficient,
                                                                               defocus)
        if expanded_scene is None:
            self.simulation.main_segments = self.simulation.main_segments.reindex(index=self.simulation.main_segments.index[::-1])
            expanded_scene, expanded_scene_no_cells, expanded_mask = get_OPL_image(scene, mask,
                                                                                   media_multiplier, cell_multiplier,
                                                                                   device_multiplier,
                                                                                   y_border_expansion_coefficient,
                                                                                   x_border_expansion_coefficient,
                                                                                   defocus)
        return expanded_scene, expanded_scene_no_cells, expanded_mask

    def optimise_synth_image(self, manual_update):

        self.real_media_mean = self.real_resize[np.where(self.media_label.data)].mean()
        self.real_cell_mean = self.real_resize[np.where(self.cell_label.data)].mean()
        self.real_device_mean = self.real_resize[np.where(self.device_label.data)].mean()
        self.real_means = np.array((self.real_media_mean, self.real_cell_mean, self.real_device_mean))

        self.real_media_var = self.real_resize[np.where(self.media_label.data)].var()
        self.real_cell_var = self.real_resize[np.where(self.cell_label.data)].var()
        self.real_device_var = self.real_resize[np.where(self.device_label.data)].var()
        self.real_vars = np.array((self.real_media_var, self.real_cell_var, self.real_device_var))

        self.image_params = (self.real_media_mean, self.real_cell_mean, self.real_device_mean, self.real_means, self.real_media_var, self.real_cell_var, self.real_device_var, self.real_vars)



        self.params = interactive(
            self.generate_test_comparison,
            {'manual': manual_update},
            media_multiplier=(-300, 300, 1),
            cell_multiplier=(-30, 30, 0.01),
            device_multiplier=(-300, 300, 1),
            sigma=(self.PSF.min_sigma, self.PSF.min_sigma * 20, self.PSF.min_sigma / 20),
            scene_no=(0, len(self.simulation.OPL_scenes) - 1, 1),
            noise_var=(0, 0.01, 0.0001),
            scale=fixed(self.PSF.scale),
            match_fourier=[True, False],
            match_histogram=[True, False],
            match_noise=[True, False],
            offset=fixed(self.simulation.offset),
            main_segments=fixed(self.simulation.main_segments),
            debug_plot=fixed(True),
            scenes=fixed(self.simulation.OPL_scenes),
            kernel=fixed(self.PSF),
            resize_amount=fixed(self.simulation.resize_amount),
            real_image=fixed(self.real_image),
            image_params=fixed(self.image_params),
            x_border_expansion_coefficient=fixed(self.x_border_expansion_coefficient),
            y_border_expansion_coefficient=fixed(self.y_border_expansion_coefficient),
            defocus=(0, 20, 0.1)
        )

        return self.params








def generate_training_data(interactive_output, sample_amount, randomise_hist_match, randomise_noise_match, sim_length,
                           burn_in, n_samples, save_dir, in_series=False, seed=False):
    """
    Generates the training data from a Jupyter interactive output of generate_test_comparison

    Parameters
    ----------
    interactive_output : ipywidgets.widgets.interaction.interactive
        The slider object generated by :func:`~SyMBac.optimisation.manual_optimise` after you have finished tweaking parameters
    sample_amount : float
        The percentage sampling variance (drawn from a uniform distribution) to vary intensities by. For example, a
        sample_amount of 0.05 will randomly sample +/- 5% above and below the chosen intensity for cells,
        media and device. Can be used to create a little bit of variance in the final training data.
    randomise_hist_match : bool
        If true, histogram matching is randomly turned on and off each time a training sample is generated
    randomise_noise_match : bool
        If true, noise matching is randomly turned on and off each time a training sample is generated
    sim_length : int
        the length of the simulation which was run
    burn_in : int
        Number of frames to wait before generating training data. Can be used to ignore the start of the simulation
        where the trench only has 1 cell in it.
    n_samples : int
        The number of training images to generate
    save_dir : str
        The save directory of the training data
    in_series : bool
        Whether the images should be randomly sampled, or rendered in the order that the simulation was run in.
    seed : float
        Optional arg, if specified then the numpy random seed will be set for the rendering, allows reproducible rendering results.

    """
    if seed:
        np.random.seed(seed)
    media_multiplier, cell_multiplier, device_multiplier, sigma, scene_no, scale, match_fourier, match_histogram, \
    match_noise, offset, debug_plot, noise_var, main_segments, scenes, kernel_params, resize_amount, real_image, \
    image_params, error_params, x_border_expansion_coefficient, y_border_expansion_coefficient, fluorescence, \
    fluo_3D, camera_noise, camera_params, defocus = list(
        interactive_output.kwargs.values())
    debug_plot = False
    try:
        os.mkdir(save_dir)
    except:
        pass
    try:
        os.mkdir(save_dir + "/convolutions")
    except:
        pass
    try:
        os.mkdir(save_dir + "/masks")
    except:
        pass

    current_file_num = len(os.listdir(save_dir + "/convolutions"))

    def generate_samples(z):
        np.random.seed(0)
        _media_multiplier = np.random.uniform(1 - sample_amount, 1 + sample_amount) * media_multiplier
        np.random.seed(1)
        _cell_multiplier = np.random.uniform(1 - sample_amount, 1 + sample_amount) * cell_multiplier
        np.random.seed(0)
        _device_multiplier = np.random.uniform(1 - sample_amount, 1 + sample_amount) * device_multiplier
        np.random.seed(1)
        _sigma = np.random.uniform(1 - sample_amount, 1 + sample_amount) * sigma
        if in_series:
            _scene_no = burn_in + z % (sim_length - 2)
            # can maybe re-run run_simulation and draw_scene when this loops back to 0
        else:
            _scene_no = np.random.randint(burn_in, sim_length - 2)
        if randomise_hist_match:
            _match_histogram = np.random.choice([True, False])
        else:
            _match_histogram = match_histogram
        if randomise_noise_match:
            _match_noise = np.random.choice([True, False])
        else:
            _match_noise = match_noise

        syn_image, mask = generate_test_comparison(_media_multiplier, _cell_multiplier, _device_multiplier, _sigma,
                                                   _scene_no, scale, match_fourier, _match_histogram, _match_noise,
                                                   offset, debug_plot, noise_var, main_segments, scenes, kernel_params,
                                                   resize_amount, real_image, image_params, error_params,
                                                   x_border_expansion_coefficient, y_border_expansion_coefficient,
                                                   fluorescence, fluo_3D, camera_noise, camera_params, defocus)

        syn_image = Image.fromarray(skimage.img_as_uint(rescale_intensity(syn_image)))
        syn_image.save("{}/convolutions/synth_{}.tif".format(save_dir, str(z).zfill(5)))

        if (cell_multiplier == 0) or (cell_multiplier == 0.0):
            mask = np.zeros(mask.shape)
            mask = Image.fromarray(mask.astype(np.uint8))
            mask.save("{}/masks/synth_{}.tif".format(save_dir, str(z).zfill(5)))
        else:
            mask = Image.fromarray(mask.astype(np.uint8))
            mask.save("{}/masks/synth_{}.tif".format(save_dir, str(z).zfill(5)))
            ## TODO: change parallel if not using GPU

    Parallel(n_jobs=1)(delayed(generate_samples)(z) for z in
                       tqdm(range(current_file_num, n_samples + current_file_num), desc="Sample generation"))


# from https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
def cart2pol(x, y):
    phi = np.arctan2(y, x)
    rho = np.sqrt(x ** 2 + y ** 2)
    return phi, rho

def pol2cart(phi, rho):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def sfMatch(images, rescaling=0, tarmag=None):
    assert type(images) == type([]), 'The input must be a list.'

    numin = len(images)
    xs, ys = images[1].shape
    angs = np.zeros((xs, ys, numin))
    mags = np.zeros((xs, ys, numin))
    for x in range(numin):
        if len(images[x].shape) == 3:
            images[x] = rgb2gray(images[x])
        im1 = images[x] / 255
        xs1, ys1 = im1.shape
        assert (xs == xs1) and (ys == ys1), 'All images must have the same size.'
        fftim1 = fft.fftshift(fft.fft2(im1))
        angs[:, :, x], mags[:, :, x] = cart2pol(np.real(fftim1), np.imag(fftim1))

    if tarmag is None:
        tarmag = np.mean(mags, 2)

    xt, yt = tarmag.shape
    assert (xs == xt) and (ys == yt), 'The target spectrum must have the same size as the images.'
    f1 = np.linspace(-ys / 2, ys / 2 - 1, ys)
    f2 = np.linspace(-xs / 2, xs / 2 - 1, xs)
    XX, YY = np.meshgrid(f1, f2)
    t, r = cart2pol(XX, YY)
    if xs % 2 == 1 or ys % 2 == 1:
        r = np.round(r) - 1
    else:
        r = np.round(r)
    output_images = []
    for x in range(numin):
        fftim = mags[:, :, x]
        a = fftim.T.ravel()
        accmap = r.T.ravel() + 1
        a2 = tarmag.T.ravel()
        en_old = np.array(
            [np.sum([a[x] for x in y]) for y in [list(np.where(accmap == z)) for z in np.unique(accmap).tolist()]])
        en_new = np.array(
            [np.sum([a2[x] for x in y]) for y in [list(np.where(accmap == z)) for z in np.unique(accmap).tolist()]])
        coefficient = en_new / en_old
        cmat = coefficient[(r).astype(int)]  # coefficient[r+1]
        cmat[r > np.floor(np.max((xs, ys)) / 2)] = 0
        newmag = fftim * cmat
        XX, YY = pol2cart(angs[:, :, x], newmag)
        new = XX + YY * complex(0, 1)
        output = np.real(fft.ifft2(fft.ifftshift(new)))
        if rescaling == 0:
            output = (output * 255)
        output_images.append(output)
    if rescaling != 0:
        output_images = rescale_shine(output_images, rescaling)
    return output_images

def rescale_shine(images, option=1):
    assert type(images) == type([]), 'The input must be a list.'
    assert option == 1 or option == 2, "Invalid rescaling option"
    numin = len(images)
    brightests = np.zeros((numin, 1))
    darkests = np.zeros((numin, 1))
    for n in range(numin):
        if len(images[n].shape) == 3:
            images[n] = rgb2gray(images[n])
        brightests[n] = np.max(images[n])
        darkests[n] = np.min(images[n])
    the_brightest = np.max(brightests)
    the_darkest = np.min(darkests)
    avg_brightest = np.mean(brightests)
    avg_darkest = np.mean(darkests)
    output_images = []
    for m in range(numin):
        if option == 1:
            rescaled = (images[m] - the_darkest) / (the_brightest - the_darkest) * 255
        else:  # option == 2:
            rescaled = (images[m] - avg_darkest) / (avg_brightest - avg_darkest) * 255
        output_images.append(rescaled.astype(np.uint8))
    return output_images

def lumMatch(images, mask=None, lum=None):
    assert type(images) == type([]), 'The input must be a list.'
    assert (mask is None) or type(mask) == type([]), 'The input mask must be a list.'

    numin = len(images)
    if (mask is None) and (lum is None):
        M = 0;
        S = 0
        for im in range(numin):
            if len(images[im].shape) == 3:
                images[im] = rgb2gray(images[im])
            M = M + np.mean(images[im])
            S = S + np.std(images[im])
        M = M / numin
        S = S / numin
        output_images = []
        for im in range(numin):
            im1 = copy.deepcopy(images[im])
            if np.std(im1) != 0:
                im1 = (im1 - np.mean(im1)) / np.std(im1) * S + M
            else:
                im1[:, :] = M
            output_images.append(im1)
    elif (mask is None) and (lum is not None):
        M = 0
        S = 0
        for im in range(numin):
            if len(images[im].shape) == 3:
                images[im] = rgb2gray(images[im])
            M = lum[0]
            S = lum[1]
        M = M / numin
        S = S / numin
        output_images = []
        for im in range(numin):
            im1 = copy.deepcopy(images[im])
            if np.std(im1) != 0:
                im1 = (im1 - np.mean(im1)) / np.std(im1) * S + M
            else:
                im1[:, :] = M
            output_images.append(im1)
    elif (mask is not None) and (lum is None):
        M = 0
        S = 0
        for im in range(numin):
            if len(images[im].shape) == 3:
                images[im] = rgb2gray(images[im])
            im1 = images[im]
            assert len(images) == len(mask), "The inputs must have the same length"
            m = mask[im]
            assert m.size == images[im].size, "Image and mask are not the same size"
            assert np.sum(m == 1) > 0, 'The mask must contain some ones.'
            M = M + np.mean(im1[m == 1])
            S = S + np.mean(im1[m == 1])
        M = M / numin
        S = S / numin
        output_images = []
        for im in range(numin):
            im1 = images[im]
            if type(mask) == type([]):
                m = mask[im]
            if np.std(im1[m == 1]):
                im1[m == 1] = (im1[m == 1] - np.mean(im1[m == 1])) / np.std(im1[m == 1]) * S + M
            else:
                im1[m == 1] = M
            output_images.append(im1)
    elif (mask is not None) and (lum is not None):
        M = lum[0]
        S = lum[1]
        output_images = []
        for im in range(numin):
            if len(images[im].shape) == 3:
                images[im] = rgb2gray(images[im])
            im1 = images[im]
            if len(mask) == 0:
                if np.std(im1) != 0.0:
                    im1 = (im1 - np.mean(im1)) / np.std(im1) * S + M
                else:
                    im1[:, :] = M
            else:
                if type(mask) == type([]):
                    assert len(images) == len(mask), "The inputs must have the same length"
                    m = mask[im]
                assert m.size == images[im].size, "Image and mask are not the same size"
                assert np.sum(m == 1) > 0, 'The mask must contain some ones.'
                if np.std(im1[m == 1]) != 0.0:
                    im1[m == 1] = (im1[m == 1] - np.mean(im1[m == 1])) / np.std(im1[m == 1]) * S + M
                else:
                    im1[m == 1] = M
            output_images.append(im1)
    return output_images