import pyglet
from matplotlib import pyplot as plt
from skimage.transform import rescale, rotate
from skimage.util import random_noise
from joblib import Parallel, delayed
from skimage.morphology import opening
from tqdm import tqdm
import pandas as pd
from skimage import draw
from skimage.exposure import match_histograms, rescale_intensity
from SyMBac.PSF import get_fluorescence_kernel, get_phase_contrast_kernel
import os
import skimage
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter
import numpy as np
from skimage.color import rgb2gray
from numpy import fft
from PIL import Image
import copy
from cupyx.scipy.ndimage import convolve as cuconvolve
from SyMBac.cell import Cell
from SyMBac.general_drawing import convolve_rescale, make_images_same_shape, perc_diff, raster_cell, transform_func
from SyMBac.cell_simulation import create_space, step_and_update
from SyMBac.trench_geometry import trench_creator
import psfmodels as psfm
import cupy as cp

def generate_PC_OPL(main_segments, offset, scene, mask, media_multiplier, cell_multiplier, device_multiplier,
                    y_border_expansion_coefficient, x_border_expansion_coefficient, fluorescence, defocus):
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

    def get_OPL_image(main_segments, offset, scene, mask, media_multiplier, cell_multiplier, device_multiplier,
                      y_border_expansion_coefficient, x_border_expansion_coefficient, fluorescence, defocus):
        segment_1_top_left = (0 + offset, int(main_segments.iloc[0]["bb"][0] + offset))
        segment_1_bottom_right = (
            int(main_segments.iloc[0]["bb"][3] + offset), int(main_segments.iloc[0]["bb"][2] + offset))

        segment_2_top_left = (0 + offset, int(main_segments.iloc[1]["bb"][0] + offset))
        segment_2_bottom_right = (
            int(main_segments.iloc[1]["bb"][3] + offset), int(main_segments.iloc[1]["bb"][2] + offset))

        if fluorescence:
            test_scene = np.zeros(scene.shape)
            media_multiplier = -1 * device_multiplier
        else:
            test_scene = np.zeros(scene.shape) + device_multiplier
            rr, cc = draw.rectangle(start=segment_1_top_left, end=segment_1_bottom_right, shape=test_scene.shape)
            test_scene[rr, cc] = 1 * media_multiplier
            rr, cc = draw.rectangle(start=segment_2_top_left, end=segment_2_bottom_right, shape=test_scene.shape)
            test_scene[rr, cc] = 1 * media_multiplier
            circ_midpoint_y = (segment_1_top_left[1] + segment_2_bottom_right[1]) / 2
            radius = (segment_1_top_left[1] - offset - (segment_2_bottom_right[1] - offset)) / 2
            circ_midpoint_x = (offset) + radius

            rr, cc = draw.rectangle(start=segment_2_top_left, end=(circ_midpoint_x, segment_1_top_left[1]),
                                    shape=test_scene.shape)
            test_scene[rr.astype(int), cc.astype(int)] = 1 * media_multiplier
            rr, cc = draw.disk(center=(circ_midpoint_x, circ_midpoint_y), radius=radius, shape=test_scene.shape)
            rr_semi = rr[rr < (circ_midpoint_x + 1)]
            cc_semi = cc[rr < (circ_midpoint_x + 1)]
            test_scene[rr_semi, cc_semi] = device_multiplier
        no_cells = copy.deepcopy(test_scene)

        test_scene += scene * cell_multiplier
        if fluorescence:
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
                                            int(no_cells.shape[1] * x_border_expansion_coefficient))) + media_multiplier
        expanded_scene_no_cells[expanded_scene_no_cells.shape[0] - no_cells.shape[0]:,
        int(expanded_scene_no_cells.shape[1] / 2 - int(test_scene.shape[1] / 2)):int(
            expanded_scene_no_cells.shape[1] / 2 - int(test_scene.shape[1] / 2)) + no_cells.shape[1]] = no_cells
        if fluorescence:
            expanded_scene = np.zeros((int(test_scene.shape[0] * y_border_expansion_coefficient),
                                       int(test_scene.shape[1] * x_border_expansion_coefficient)))
            expanded_scene[expanded_scene.shape[0] - test_scene.shape[0]:,
            int(expanded_scene.shape[1] / 2 - int(test_scene.shape[1] / 2)):int(
                expanded_scene.shape[1] / 2 - int(test_scene.shape[1] / 2)) + test_scene.shape[1]] = test_scene
        else:
            expanded_scene = np.zeros((int(test_scene.shape[0] * y_border_expansion_coefficient),
                                       int(test_scene.shape[1] * x_border_expansion_coefficient))) + media_multiplier
            expanded_scene[expanded_scene.shape[0] - test_scene.shape[0]:,
            int(expanded_scene.shape[1] / 2 - int(test_scene.shape[1] / 2)):int(
                expanded_scene.shape[1] / 2 - int(test_scene.shape[1] / 2)) + test_scene.shape[1]] = test_scene

        expanded_mask = np.zeros((int(test_scene.shape[0] * y_border_expansion_coefficient),
                                  int(test_scene.shape[1] * x_border_expansion_coefficient)))
        expanded_mask[expanded_mask.shape[0] - test_scene.shape[0]:,
        int(expanded_mask.shape[1] / 2 - int(test_scene.shape[1] / 2)):int(
            expanded_mask.shape[1] / 2 - int(test_scene.shape[1] / 2)) + test_scene.shape[1]] = mask_resized

        return expanded_scene, expanded_scene_no_cells, expanded_mask

    expanded_scene, expanded_scene_no_cells, expanded_mask = get_OPL_image(main_segments, offset, scene, mask,
                                                                           media_multiplier, cell_multiplier,
                                                                           device_multiplier,
                                                                           y_border_expansion_coefficient,
                                                                           x_border_expansion_coefficient, fluorescence,
                                                                           defocus)
    if expanded_scene is None:
        main_segments = main_segments.reindex(index=main_segments.index[::-1])
        expanded_scene, expanded_scene_no_cells, expanded_mask = get_OPL_image(main_segments, offset, scene, mask,
                                                                               media_multiplier, cell_multiplier,
                                                                               device_multiplier,
                                                                               y_border_expansion_coefficient,
                                                                               x_border_expansion_coefficient,
                                                                               fluorescence, defocus)
    return expanded_scene, expanded_scene_no_cells, expanded_mask

def generate_test_comparison(media_multiplier=75, cell_multiplier=1.7, device_multiplier=29, sigma=8.85, scene_no=-1,
                             scale=None, match_fourier=False, match_histogram=True, match_noise=False, offset=30,
                             debug_plot=False, noise_var=0.001, main_segments=None, scenes=None, kernel_params=None,
                             resize_amount=None, real_image=None, image_params=None, error_params=None,
                             x_border_expansion_coefficient=None, y_border_expansion_coefficient=None,
                             fluorescence=False, fluo_3D=False, camera_noise=False, camera_params=None, defocus=3.0):
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
    main_segments : list
        List of trench segment properties, output of get_trench_segments function
    scenes : list(2D numpy array)
        A list of the previously rendered scene mask pairs
    kernel_params : tuple
        A tuple of kernel parameters in this order: (R,W,radius,scale,NA,n,sigma,λ)
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
    y_border_expansioon_coefficient : int
        Another offset-like argument. Multiplies the size of the image on each side by this value. 3 is a good starting
        value because you want the image to be relatively larger than the PSF which you are convolving over it.
    x_border_expansioon_coefficient : int
        Another offset-like argument. Multiplies the size of the image on each side by this value. 3 is a good starting
        value because you want the image to be relatively larger than the PSF which you are convolving over it.
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

    expanded_scene, expanded_scene_no_cells, expanded_mask = generate_PC_OPL(
        main_segments=main_segments,
        offset=offset,
        scene=scenes[scene_no][0],
        mask=scenes[scene_no][1],
        media_multiplier=media_multiplier,
        cell_multiplier=cell_multiplier,
        device_multiplier=device_multiplier,
        x_border_expansion_coefficient=x_border_expansion_coefficient,
        y_border_expansion_coefficient=y_border_expansion_coefficient,
        fluorescence=fluorescence,
        defocus=defocus
    )

    R, W, radius, scale, NA, n, _, λ = kernel_params

    real_media_mean, real_cell_mean, real_device_mean, real_means, real_media_var, real_cell_var, real_device_var, real_vars = image_params
    mean_error, media_error, cell_error, device_error, mean_var_error, media_var_error, cell_var_error, device_var_error = error_params


    
    if fluorescence and fluo_3D: #Full 3D PSF model
        def generate_deviation_from_CL(centreline, thickness):
            return np.arange(thickness)+centreline - int(np.ceil(thickness/2))
        def gen_3D_coords_from_2D(centreline, thickness):
            return np.where(test_cells==thickness) + (generate_deviation_from_CL(centreline, thickness),)

        volume_shape = expanded_scene.shape[0:] + (int(expanded_scene.max()),)
        test_cells = np.round(expanded_scene)
        centreline = int(expanded_scene.max()/2)
        cells_3D = np.zeros(volume_shape)
        for t in range(int(expanded_scene.max()*2)):
            test_coords = gen_3D_coords_from_2D(centreline, t)
            for x, y in zip(test_coords[0], (test_coords[1])):
                for z in test_coords[2]:
                    cells_3D[x,y,z] = 1
        cells_3D = np.moveaxis(cells_3D, -1, 0)
        psf = psfm.make_psf(volume_shape[2], radius*2, dxy=scale, dz=scale, pz=0, ni=n, wvl=λ, NA = NA)
        convolved = np.zeros(cells_3D.shape)
        for x in range(len(cells_3D)):
            temp_conv = cuconvolve(cp.array(cells_3D[x]), cp.array(psf[x])).get()
            convolved[x] = temp_conv
        convolved = convolved.sum(axis=0)
        convolved = rescale(convolved, 1 / resize_amount, anti_aliasing=False)
        convolved = rescale_intensity(convolved.astype(np.float32), out_range=(0, 1))
    else:
        if fluorescence:
            kernel = get_fluorescence_kernel(radius=radius, scale=scale, NA=NA, n=n, Lambda=λ)[0]
            if defocus > 0:
                kernel = gaussian_filter(kernel, defocus, mode="reflect")
        else:
            kernel = get_phase_contrast_kernel(R=R, W=W, radius=radius, scale=scale, NA=NA, n=n, sigma=sigma, λ=λ)
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
    
    if camera_noise: #Camera noise simulation
        baseline, sensitivity, dark_noise = camera_params
        rng = np.random.default_rng(2)
        matched = matched/(matched.max()/real_image.max()) / sensitivity
        matched = rng.poisson(matched)
        noisy_img = matched + rng.normal(loc = baseline, scale=dark_noise, size=matched.shape)
    else: #Ad hoc noise mathcing
        noisy_img = random_noise(rescale_intensity(matched), mode="poisson")
        noisy_img = random_noise(rescale_intensity(noisy_img), mode="gaussian", mean=0, var=noise_var, clip=False)

    if match_noise:
        noisy_img = match_histograms(noisy_img, real_resize, multichannel=False)
    else:
        pass
    noisy_img = rescale_intensity(noisy_img.astype(np.float32), out_range=(0, 1))

    ## getting the cell mask to the right shape
    expanded_mask_resized = rescale(expanded_mask, 1 / resize_amount, anti_aliasing=False, preserve_range=True, order=0)
    if len(np.unique(expanded_mask_resized)) > 2:
        _, expanded_mask_resized_reshaped = make_images_same_shape(real_image, expanded_mask_resized, rescale_int=False)
    else:
        _, expanded_mask_resized_reshaped = make_images_same_shape(real_image, expanded_mask_resized, rescale_int=True)

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
    if fluorescence:
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
        ax3.hlines(0, ax3.get_xlim()[0], ax3.get_xlim()[1], color = "k", linestyles="dotted")
        ax4.plot(mean_var_error)
        ax4.plot(media_var_error)
        ax4.plot(cell_var_error)
        ax4.plot(device_var_error)
        ax4.legend(["Mean error", "Media error", "Cell error", "Device error"])
        ax4.set_title("Variance Error")
        ax4.hlines(0, ax4.get_xlim()[0], ax4.get_xlim()[1], color = "k", linestyles="dotted")
        fig.tight_layout()
        plt.show()
        plt.close()
    else:
        return noisy_img, expanded_mask_resized_reshaped.astype(int)

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
