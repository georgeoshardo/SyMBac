import numpy as np
import pandas as pd
from scipy.special import jv, jve
from skimage.util import random_noise
import raster_geometry as rg
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
import pickle
import sys
from skimage.transform import rescale, resize, downscale_local_mean
import itertools
from joblib import Parallel, delayed
from skimage.morphology import opening
from PIL import Image       
import pymunk
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
from skimage import draw
#import napari
from itertools import combinations
from SyMBac import PSF
from matplotlib_scalebar.scalebar import ScaleBar
import tifffile
from skimage.exposure import match_histograms
from scipy.optimize import dual_annealing, shgo
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import basinhopping
#import image_similarity_measures
#from image_similarity_measures.quality_metrics import rmse, psnr, fsim, issm, sre, sam, uiq
#sys.path.insert(0,'/home/georgeos/Documents/GitHub/SYMPTOMM2')
from SyMBac.general_drawing import *
from SyMBac.phase_contrast_drawing import *
from SyMBac.scene_functions import *
from SyMBac.trench_geometry import *
from SyMBac.PSF import *
import os
import skimage
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter

##From here, prototyping phase contrast
def get_trench_segments(space):
    """    
    A function which extracts the rigid body trench objects from the pymunk space object. Space object should be passed from the return value of the run_simulation() function
    
    Returns
    -------
    List of trench segment properties, later used to draw the trench.
    """
    trench_shapes = []
    for shape, body in zip(space.shapes, space.bodies):
        if body.body_type == 2:
            trench_shapes.append(shape)
            
    trench_segment_props = []
    for x in trench_shapes:
        trench_segment_props.append([x.bb,x.area, x.a, x.b])   

    trench_segment_props = pd.DataFrame(trench_segment_props)
    trench_segment_props.columns = ["bb", "area", "a", "b"]
    main_segments = trench_segment_props.sort_values("area",ascending=False).iloc[0:2]
    return main_segments

    
def return_intersection_between_image_hists(img1, img2, bins):
    hist_1, _ = np.histogram(img1.flatten()/img1.flatten().max(), bins=bins)
    hist_2, _ = np.histogram(img2.flatten()/img2.flatten().max(), bins=bins)
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection


def similarity_objective_function(z, ret_tuple=False):
    real_image = tifffile.imread(
        "/home/georgeos/Storage/Dropbox (Cambridge University)/PhD_Georgeos_Hardo/ML_based_segmentation_results/40x_Ph2_test_1.5/top_trenches_PC/trench_{}/T_{}.tif".format(
            str(np.random.randint(1, 56)).zfill(2),
            str(np.random.randint(20, 25)).zfill(3)))

    real_image = real_image.astype(np.float64) / np.max(real_image)

    σ, media_multiplier, cell_multiplier, device_multiplier, trench_length, trench_width, cell_max_length, cell_width = z
    print(z)

    cell_timeseries, space = run_simulation(trench_length, trench_width, cell_max_length, cell_width)
    main_segments = get_trench_segments(space)
    ID_props = generate_curve_props(cell_timeseries)
    cell_timeseries_properties = Parallel(n_jobs=14)(
        delayed(gen_cell_props_for_draw)(a, ID_props) for a in cell_timeseries[-5:-1])
    do_transformation = True
    scenes = Parallel(n_jobs=14)(
        delayed(draw_scene)(cell_properties, do_transformation) for cell_properties in tqdm(cell_timeseries_properties))

    # expanded_scene = generate_PC_OPL(media_multiplier,cell_multiplier,device_multiplier)
    OPL_scenes = [
        generate_PC_OPL(main_segments, offset, scene[1], media_multiplier, cell_multiplier, device_multiplier) for
        scene in scenes]

    kernel = get_phase_contrast_kernel(R, W, 50, scale, 5, σ, 0.6)
    OPL_scenes_convolved = np.array([convolve_rescale(OPL_scene, kernel, 1) for OPL_scene in OPL_scenes])

    # convolved_image = random_noise(convolved_image, mode="gaussian", mean=5,var=σ2,clip=False)
    OPL_scenes_convolved = np.array(
        [match_histograms(OPL_scene_convolved, real_image, multichannel=False) for OPL_scene_convolved in
         OPL_scenes_convolved])
    all_objs = [get_similarity_metrics(*make_images_same_shape(real_image, OPL_scene_convolved)) for OPL_scene_convolved
                in OPL_scenes_convolved]
    objs = np.mean(all_objs, axis=0)
    # convolved_image = resize(convolved_image,real_image.shape,clip=False,preserve_range=False,anti_aliasing=None)
    # ssim_real = ssim(convolved_image, real_image)
    # intersection = return_intersection_between_image_hists(convolved_image, real_image, 100)
    # sims
    # convolved_image.shape += (1,)
    # real_image.shape += (1,)
    # _fsim = fsim(convolved_image,real_image)
    # _issm = issm(convolved_image,real_image)
    # _sam = sam(convolved_image,real_image)
    # _sre = sre(convolved_image,real_image)
    # objs = [ssim_real, 0.5*intersection, _fsim, _issm, _sam, _sre/20]
    if ret_tuple == False:
        return -np.linalg.norm(objs)
    else:
        return objs, OPL_scenes_convolved
    
def run_simulation(trench_length, trench_width, cell_max_length, cell_width, sim_length, pix_mic_conv, gravity, phys_iters, max_length_var, width_var, save_dir):
    
    """
    Runs the rigid body simulation of bacterial growth based on a variety of parameters. Opens up a Pyglet window to display the animation in real-time. If the simulation looks bad to your eye, restart the kernel and rerun the simulation. There is currently a bug where if you try to rerun the simulation in the same kernel, it will be extremely slow.
    
    Parameters
    ----------
    
    trench_length : float
        Length of a mother machine trench (micron)
    trench_width : float
        Width of a mother machine trench (micron)
    cell_max_length : float
        Maximum length a cell can reach before dividing (micron)
    cell_width : float
        the average cell width in the simmulation (micron)
    pix_mic_conv : float
        The micron/pixel size of the image
    gravity : float
        Pressure forcing cells into the trench. Typically left at zero, but can be varied if cells start to fall into each other or if the simulation behaves strangely.
    phys_iters : int
        Number of physics iterations per simulation frame. Increase to resolve collisions if cells are falling into one another, but decrease if cells begin to repel one another too much (too high a value causes cells to bounce off each other very hard). 20 is a good starting point
    max_length_var : float
        Variance of the maximum cell length
    width_var : float
        Variance of the maximum cell width
    save_dir : str
        Location to save simulation outupt
        
    Returns
    -------
    cell_timeseries : lists
        A list of parameters for each cell, such as length, width, position, angle, etc. All used in the drawing of the scene later
    space : a pymunk space object
        Contains the rigid body physics objects which are the cells.
    """
    
    
    space = create_space()
    space.gravity = 0, gravity # arbitrary units, negative is toward trench pole
    dt = 1/100 #time-step per frame
    pix_mic_conv = 1/pix_mic_conv # micron per pixel
    scale_factor = pix_mic_conv * 3 # resolution scaling factor 

    trench_length = trench_length*scale_factor
    trench_width = trench_width*scale_factor
    trench_creator(trench_width,trench_length,(35,0),space) # Coordinates of bottom left corner of the trench

    cell1 = Cell(
        length = cell_max_length*scale_factor,  
        width = cell_width*scale_factor, 
        resolution = 60, 
        position = (20+35,40), 
        angle = 0.8, 
        space = space,
        dt = 1/60,
        growth_rate_constant = 1,
        max_length = cell_max_length*scale_factor,
        max_length_mean =cell_max_length*scale_factor,
        max_length_var = max_length_var*np.sqrt(scale_factor),
        width_var = width_var*np.sqrt(scale_factor),
        width_mean = cell_width*scale_factor
    )
    
    
    window = pyglet.window.Window(700, 700, "SyMBac", resizable=True)
    options = DrawOptions()
    @window.event
    def on_draw():
        window.clear()
        space.debug_draw(options)

    global cell_timeseries
    global x    
    
    
    try:
        del cell_timeseries
    except:
        pass
    try:
        del x
    except:
        pass
    
    x = [0]
    cell_timeseries = []
    cells = [cell1]
    pyglet.clock.schedule_interval(step_and_update, interval=dt, cells=cells, space=space, phys_iters=phys_iters ,ylim=trench_length, cell_timeseries = cell_timeseries,x=x,sim_length=sim_length, save_dir=save_dir)
    pyglet.app.run()
    #window.close()
    #phys_iters = phys_iters
    #for x in tqdm(range(sim_length+250),desc="Simulation Progress"):
    #    cells = step_and_update(dt=dt, cells=cells, space=space, phys_iters=phys_iters,ylim=trench_length*1.1, cell_timeseries = cell_timeseries, x=x, sim_length = sim_length, save_dir = save_dir)
    #    if x > 250:
    #        cell_timeseries.append(deepcopy(cells))
    return cell_timeseries, space

def get_similarity_metrics(real_image,synthetic_image):
    """ No longer used function to get a list of similarity metrics between two images, may be re-purposed later"""    
    synthetic_image = match_histograms(synthetic_image, real_image, multichannel=False)
    synthetic_image = resize(synthetic_image,real_image.shape,clip=False,preserve_range=False,anti_aliasing=None)
    synthetic_image = synthetic_image/np.max(synthetic_image)
    ssim_real = ssim(synthetic_image, real_image)
    intersection = return_intersection_between_image_hists(synthetic_image, real_image, 100)
    #sims 
    synthetic_image_ = deepcopy(synthetic_image)
    synthetic_image_.shape += (1,)
    
    real_image_ = deepcopy(real_image)
    real_image_.shape += (1,)
    _fsim = fsim(synthetic_image_,real_image_)
    _issm = issm(synthetic_image_,real_image_)
    _sam = sam(synthetic_image_,real_image_)
    _sre = sre(synthetic_image_,real_image_)
    objs = [ssim_real, 0.5*intersection, _fsim, _issm, _sam, _sre/20]
    return objs

def generate_PC_OPL(main_segments, offset, scene, mask, media_multiplier,cell_multiplier,device_multiplier, y_border_expansion_coefficient, x_border_expansion_coefficient, fluorescence, defocus):
    """
    Takes a scene drawing, adds the trenches and colours all parts of the image to generate a first-order phase contrast image, uncorrupted (unconvolved) by the phase contrat optics. Also has a fluorescence parameter to quickly switch to fluorescence if you want. 
    
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
    y_border_expansioon_coefficient : int
        Another offset-like argument. Multiplies the size of the image on each side by this value. 3 is a good starting value because you want the image to be relatively larger than the PSF which you are convolving over it.
    x_border_expansioon_coefficient : int
        Another offset-like argument. Multiplies the size of the image on each side by this value. 3 is a good starting value because you want the image to be relatively larger than the PSF which you are convolving over it.
    fluorescence : bool
        If true converts image to a fluorescence (hides the trench and swaps to the fluorescence PSF). 
    defocus : float
        Simulated optical defocus by convolving the kernel with a 2D gaussian of radius defocus.
        
    Returns
    -------
    expanded_scene : 2D numpy array
        A large (expanded on x and y axis) image of cells in a trench, but unconvolved. (The raw PC image before convolution)
    expanded_scene_no_cells : 2D numpy array
        Same as expanded_scene, except with the cells removed (this is necessary for later intensity tuning)
    expanded_mask : 2D numpy array
        The masks for the expanded scene
    """
    
    def get_OPL_image(main_segments, offset, scene, mask, media_multiplier,cell_multiplier,device_multiplier, y_border_expansion_coefficient, x_border_expansion_coefficient, fluorescence, defocus):
        segment_1_top_left = (0 + offset, int(main_segments.iloc[0]["bb"][0] + offset))
        segment_1_bottom_right = (int(main_segments.iloc[0]["bb"][3] + offset), int(main_segments.iloc[0]["bb"][2] + offset))

        segment_2_top_left = (0 + offset, int(main_segments.iloc[1]["bb"][0] + offset))
        segment_2_bottom_right = (int(main_segments.iloc[1]["bb"][3] + offset), int(main_segments.iloc[1]["bb"][2] + offset))
        
        if fluorescence:
            test_scene = np.zeros(scene.shape)
            media_multiplier = -1*device_multiplier
        else:
            test_scene = np.zeros(scene.shape) + device_multiplier
            rr, cc = draw.rectangle(start = segment_1_top_left, end = segment_1_bottom_right, shape = test_scene.shape)
            test_scene[rr,cc] = 1 * media_multiplier
            rr, cc = draw.rectangle(start = segment_2_top_left, end = segment_2_bottom_right, shape = test_scene.shape)
            test_scene[rr,cc] = 1 * media_multiplier
            circ_midpoint_y = (segment_1_top_left[1] + segment_2_bottom_right[1])/2
            radius = (segment_1_top_left[1] - offset - (segment_2_bottom_right[1] - offset))/2
            circ_midpoint_x = (offset) + radius

            rr, cc = draw.rectangle(start = segment_2_top_left, end = (circ_midpoint_x,segment_1_top_left[1]), shape = test_scene.shape)
            test_scene[rr.astype(int),cc.astype(int)] = 1 * media_multiplier
            rr, cc = draw.disk(center = (circ_midpoint_x, circ_midpoint_y), radius = radius, shape = test_scene.shape)
            rr_semi = rr[rr < (circ_midpoint_x + 1)]
            cc_semi = cc[rr < (circ_midpoint_x + 1)]
            test_scene[rr_semi,cc_semi] = device_multiplier
        no_cells = deepcopy(test_scene)
        
        
        
        test_scene += scene * cell_multiplier
        if fluorescence:
            pass
        else:
            test_scene = np.where(no_cells != media_multiplier, test_scene, media_multiplier)
        test_scene = test_scene[segment_2_top_left[0]:segment_1_bottom_right[0],segment_2_top_left[1]:segment_1_bottom_right[1]]
        
        mask = np.where(no_cells != media_multiplier, mask, 0)
        mask_resized = mask[segment_2_top_left[0]:segment_1_bottom_right[0],segment_2_top_left[1]:segment_1_bottom_right[1]]
        
        no_cells = no_cells[segment_2_top_left[0]:segment_1_bottom_right[0],segment_2_top_left[1]:segment_1_bottom_right[1]]
        expanded_scene_no_cells = np.zeros((int(no_cells.shape[0]*y_border_expansion_coefficient), int(no_cells.shape[1]*x_border_expansion_coefficient))) + media_multiplier
        expanded_scene_no_cells[expanded_scene_no_cells.shape[0] - no_cells.shape[0]:,
                                int(expanded_scene_no_cells.shape[1]/2-int(test_scene.shape[1]/2)):int(expanded_scene_no_cells.shape[1]/2-int(test_scene.shape[1]/2)) + no_cells.shape[1]] = no_cells
        if fluorescence:
            expanded_scene = np.zeros((int(test_scene.shape[0]*y_border_expansion_coefficient), int(test_scene.shape[1]*x_border_expansion_coefficient)))
            expanded_scene[expanded_scene.shape[0] - test_scene.shape[0]:,
                       int(expanded_scene.shape[1]/2-int(test_scene.shape[1]/2)):int(expanded_scene.shape[1]/2-int(test_scene.shape[1]/2)) + test_scene.shape[1]] = test_scene
        else:
            expanded_scene = np.zeros((int(test_scene.shape[0]*y_border_expansion_coefficient), int(test_scene.shape[1]*x_border_expansion_coefficient))) + media_multiplier
            expanded_scene[expanded_scene.shape[0] - test_scene.shape[0]:,
                       int(expanded_scene.shape[1]/2-int(test_scene.shape[1]/2)):int(expanded_scene.shape[1]/2-int(test_scene.shape[1]/2)) + test_scene.shape[1]] = test_scene
        
        expanded_mask = np.zeros((int(test_scene.shape[0]*y_border_expansion_coefficient), int(test_scene.shape[1]*x_border_expansion_coefficient)))
        expanded_mask[expanded_mask.shape[0] - test_scene.shape[0]:,
                      int(expanded_mask.shape[1]/2-int(test_scene.shape[1]/2)):int(expanded_mask.shape[1]/2-int(test_scene.shape[1]/2)) + test_scene.shape[1]] = mask_resized
        
        return expanded_scene, expanded_scene_no_cells, expanded_mask
    expanded_scene, expanded_scene_no_cells, expanded_mask = get_OPL_image(main_segments, offset, scene, mask, media_multiplier,cell_multiplier,device_multiplier, y_border_expansion_coefficient, x_border_expansion_coefficient, fluorescence, defocus)
    if expanded_scene is None:
        main_segments = main_segments.reindex(index=main_segments.index[::-1])
        expanded_scene, expanded_scene_no_cells, expanded_mask = get_OPL_image(main_segments, offset, scene, mask, media_multiplier,cell_multiplier,device_multiplier, y_border_expansion_coefficient, x_border_expansion_coefficient, fluorescence, defocus)
    return expanded_scene, expanded_scene_no_cells, expanded_mask

def generate_test_comparison(media_multiplier, cell_multiplier, device_multiplier, sigma, scene_no, scale, match_fourier, match_histogram, match_noise, offset, debug_plot, noise_var, main_segments, scenes, kernel_params, resize_amount, real_image, image_params, error_params, x_border_expansion_coefficient,y_border_expansion_coefficient,fluorescence,defocus):
    
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
        The upscaling factor to render the image by. E.g a resize_amount of 3 will interally render the image at 3x resolution before convolving and then downsampling the image. Values >2 are recommended.
    real_image : 2D numpy array
        A sample real image from the experiment you are trying to replicate
    image_params : tuple
        A tuple of parameters which describe the intensities and variances of the real image, in this order: (real_media_mean, real_cell_mean, real_device_mean, real_means, real_media_var, real_cell_var, real_device_var, real_vars).
    error_params : tuple
        A tuple of parameters which characterises the error between the intensities in the real image and the synthetic image, in this order: (mean_error,media_error,cell_error,device_error,mean_var_error,media_var_error,cell_var_error,device_var_error). I have given an example of their calculation in the example notebooks.
y_border_expansioon_coefficient : int
        Another offset-like argument. Multiplies the size of the image on each side by this value. 3 is a good starting value because you want the image to be relatively larger than the PSF which you are convolving over it.
    x_border_expansioon_coefficient : int
        Another offset-like argument. Multiplies the size of the image on each side by this value. 3 is a good starting value because you want the image to be relatively larger than the PSF which you are convolving over it.
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
        scene = scenes[scene_no][0],
        mask = scenes[scene_no][1],
        media_multiplier=media_multiplier,
        cell_multiplier=cell_multiplier,
        device_multiplier=device_multiplier,
        x_border_expansion_coefficient = x_border_expansion_coefficient,
        y_border_expansion_coefficient = y_border_expansion_coefficient,
        fluorescence = fluorescence,
        defocus = defocus
    )



    R,W,radius,scale,NA,n,_,λ = kernel_params
    
    
    real_media_mean, real_cell_mean, real_device_mean, real_means, real_media_var, real_cell_var, real_device_var, real_vars = image_params
    mean_error,media_error,cell_error,device_error,mean_var_error,media_var_error,cell_var_error,device_var_error = error_params
    
    if fluorescence:
        kernel = get_fluorescence_kernel(radius=radius, scale=scale, NA=NA,n=n, Lambda=λ)[0]
        if defocus > 0:
            kernel = gaussian_filter(kernel,defocus,mode="reflect")
    else:
        kernel = get_phase_contrast_kernel(R=R, W=W, radius=radius, scale=scale, NA=NA,n=n, sigma=sigma, λ=λ)
        if defocus > 0:
            kernel = gaussian_filter(kernel,defocus,mode="reflect")


    convolved = convolve_rescale(expanded_scene, kernel, 1/resize_amount, rescale_int = True)
    real_resize, expanded_resized = make_images_same_shape(real_image,convolved, rescale_int=True)
    fftim1 = fft.fftshift(fft.fft2(real_resize))
    angs, mags = cart2pol(np.real(fftim1),np.imag(fftim1))
    
    if match_fourier and not match_histogram:
        matched = sfMatch([real_resize, expanded_resized],tarmag = mags)[1]
        matched = lumMatch([real_resize,matched],None,[np.mean(real_resize),np.std(real_resize)])[1]
    else:
        matched = expanded_resized
    
    if match_histogram and match_fourier:
        matched = sfMatch([real_resize, matched],tarmag = mags)[1]
        matched = lumMatch([real_resize,matched],None,[np.mean(real_resize),np.std(real_resize)])[1]
        matched = match_histograms(matched, real_resize, multichannel=False)
    else:
        pass
    if match_histogram:
        matched = match_histograms(matched, real_resize, multichannel=False)
    else:
        pass
    
    noisy_img = random_noise(rescale_intensity(matched), mode="poisson")
    noisy_img = random_noise(rescale_intensity(noisy_img), mode="gaussian", mean=0,var=noise_var,clip=False)
    
    if match_noise:
        noisy_img = match_histograms(noisy_img, real_resize, multichannel=False)
    else:
        pass
    noisy_img = rescale_intensity(noisy_img.astype(np.float32), out_range=(0,1))
    
    ## getting the cell mask to the right shape
    expanded_mask_resized = rescale(expanded_mask, 1/resize_amount, anti_aliasing=False, preserve_range=True,order=0)
    if len(np.unique(expanded_mask_resized)) > 2:
        _, expanded_mask_resized_reshaped = make_images_same_shape(real_image,expanded_mask_resized, rescale_int=False)
    else:
        _, expanded_mask_resized_reshaped = make_images_same_shape(real_image,expanded_mask_resized, rescale_int=True)


    

    
    expanded_media_mask = rescale((expanded_scene_no_cells == device_multiplier) ^ (expanded_scene - expanded_scene_no_cells).astype(bool) , 1/resize_amount, anti_aliasing=False)
    real_resize, expanded_media_mask = make_images_same_shape(real_image,expanded_media_mask, rescale_int=True)
    just_media = expanded_media_mask * noisy_img
    
    expanded_cell_pseudo_mask = (expanded_scene - expanded_scene_no_cells).astype(bool)
    expanded_cell_pseudo_mask = rescale(expanded_cell_pseudo_mask, 1/resize_amount, anti_aliasing=False)

    real_resize, expanded_cell_pseudo_mask = make_images_same_shape(real_image,expanded_cell_pseudo_mask, rescale_int=True)
    just_cells = expanded_cell_pseudo_mask * noisy_img
    if True:
        expanded_device_mask = expanded_scene_no_cells
    else:
        expanded_device_mask = expanded_scene_no_cells == media_multiplier
    expanded_device_mask = rescale(expanded_device_mask, 1/resize_amount, anti_aliasing=False)
    real_resize, expanded_device_mask = make_images_same_shape(real_image,expanded_device_mask, rescale_int=True)
    just_device = expanded_device_mask * noisy_img
    
    
    
    
    simulated_means = np.array([just_media[np.where(just_media)].mean(), just_cells[np.where(just_cells)].mean(), just_device[np.where(just_device)].mean()])
    simulated_vars = np.array([just_media[np.where(just_media)].var(), just_cells[np.where(just_cells)].var(), just_device[np.where(just_device)].var()])

    
    
    
    if fluorescence:
        mean_error.append(perc_diff(np.mean(noisy_img),np.mean(real_resize)))
        mean_var_error.append(perc_diff(np.var(noisy_img),np.var(real_resize)))
    else:
        mean_error.append(np.mean(perc_diff(real_means, simulated_means)))
        media_error.append(perc_diff(simulated_means[0], real_media_mean))
        cell_error.append(perc_diff(simulated_means[1], real_cell_mean))
        device_error.append(perc_diff(simulated_means[2], real_device_mean))


        mean_var_error.append(np.mean(perc_diff(real_vars, simulated_vars)))
        media_var_error.append(perc_diff(simulated_vars[0], real_media_var))
        cell_var_error.append(perc_diff(simulated_vars[1], real_cell_var))
        device_var_error.append(perc_diff(simulated_vars[2], real_device_var))
    if debug_plot == True:
        fig = plt.figure(figsize=(15,5))
        ax1 = plt.subplot2grid((1,8),(0,0),colspan=1,rowspan=1)
        ax2 = plt.subplot2grid((1,8),(0,1),colspan=1,rowspan=1)
        ax3 = plt.subplot2grid((1,8),(0,2),colspan=3,rowspan=1)
        ax4 = plt.subplot2grid((1,8),(0,5),colspan=3,rowspan=1)
        ax1.imshow(noisy_img,cmap="Greys_r")
        ax1.set_title("Synthetic")
        ax1.axis("off")
        ax2.imshow(real_resize,cmap="Greys_r")
        ax2.set_title("Real")
        ax2.axis("off")
        ax3.plot(mean_error)
        ax3.plot(media_error)
        ax3.plot(cell_error)
        ax3.plot(device_error)
        ax3.legend(["Mean error", "Media error", "Cell error", "Device error"])
        ax3.set_title("Intensity Error")

        ax4.plot(mean_var_error)
        ax4.plot(media_var_error)
        ax4.plot(cell_var_error)
        ax4.plot(device_var_error)
        ax4.legend(["Mean error", "Media error", "Cell error", "Device error"])
        ax4.set_title("Variance Error")

        fig.tight_layout()
        plt.show()
        plt.close()
    else:
        return noisy_img, expanded_mask_resized_reshaped.astype(int)
    
    
def draw_scene(cell_properties, do_transformation, mask_threshold, space_size, offset, label_masks):
    """
    Draws a raw scene (no trench) of cells, and returns accompanying masks for training data.
    
    Parameters
    ----------
    Cell properties : list
        A list of cell properties for that frame
    do_transformation : bool
        True if you want cells to be bent, false and cells remain straight as in the simulation
    mask_threshold : depracated param
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
    space_size = np.array(space_size) # 1000, 200 a good value
    space = np.zeros(space_size)
    space_masks_label = np.zeros(space_size)
    space_masks_nolabel = np.zeros(space_size)    
    colour_label = [1]
    
    for properties in cell_properties:
        length, width, angle, position, freq_modif, amp_modif, phase_modif,phase_mult = properties
        length = length; width = width ; position = np.array(position) 
        x = np.array(position).astype(int)[0] + offset
        y = np.array(position).astype(int)[1] + offset
        OPL_cell = raster_cell(length = length, width=width)

        if do_transformation:
            OPL_cell_2 = np.zeros((OPL_cell.shape[0],int(OPL_cell.shape[1]*2)))
            midpoint = int(np.median(range(OPL_cell_2.shape[1])))
            OPL_cell_2[:,midpoint-int(OPL_cell.shape[1]/2):midpoint-int(OPL_cell.shape[1]/2)+OPL_cell.shape[1]] = OPL_cell
            roll_coords = np.array(range(OPL_cell_2.shape[0]))
            freq_mult = (OPL_cell_2.shape[0])
            amp_mult = OPL_cell_2.shape[1]/10
            sin_transform_cell = transform_func(amp_modif, freq_modif, phase_modif)
            roll_amounts = sin_transform_cell(roll_coords,amp_mult,freq_mult,phase_mult)
            for B in roll_coords:
                OPL_cell_2[B,:] = np.roll(OPL_cell_2[B,:], roll_amounts[B])
            OPL_cell = (OPL_cell_2)

        rotated_OPL_cell = rotate(OPL_cell,-angle,resize=True,clip=False,preserve_range=True, center=(x,y))
        cell_y, cell_x = (np.array(rotated_OPL_cell.shape)/2).astype(int)
        offset_y = rotated_OPL_cell.shape[0] - space[y-cell_y:y+cell_y,x-cell_x:x+cell_x].shape[0]
        offset_x = rotated_OPL_cell.shape[1] - space[y-cell_y:y+cell_y,x-cell_x:x+cell_x].shape[1]
        assert y > cell_y, "Cell has {} negative pixels in y coordinate, try increasing your offset".format(y - cell_y)
        assert x > cell_x, "Cell has negative pixels in x coordinate, try increasing your offset"
        space[
            y-cell_y:y+cell_y+offset_y,
            x-cell_x:x+cell_x+offset_x
        ] += rotated_OPL_cell

        def get_mask(label_masks):
            
            if label_masks:
                space_masks_label[y-cell_y:y+cell_y+offset_y,x-cell_x:x+cell_x+offset_x] += (rotated_OPL_cell > 0)*colour_label[0]
                colour_label[0] += 1
                return space_masks_label
            else:
                space_masks_nolabel[y-cell_y:y+cell_y+offset_y,x-cell_x:x+cell_x+offset_x] += (rotated_OPL_cell > 0)*1
                return space_masks_nolabel
                #space_masks = opening(space_masks,np.ones((2,11)))
            
            
        label_mask = get_mask(True).astype(int)
        nolabel_mask = get_mask(False).astype(int)
        label_mask_fixed = np.where(nolabel_mask > 1,0,label_mask)
        mask_borders = find_boundaries(label_mask_fixed,mode="thick", connectivity=2)
        space_masks = np.where(mask_borders, 0,label_mask_fixed)
        space_masks = opening(space_masks)
        if label_masks == False:
            space_masks = space_masks.astype(bool)
        space = space*space_masks.astype(bool)
    return space, space_masks

def generate_training_data(interactive_output, sample_amount, randomise_hist_match, randomise_noise_match, sim_length, burn_in, n_samples, save_dir):
    """
    Generates the training data from a Jupyter interactive output of generate_test_comparison
    
    Parameters
    ----------
    interactive_output : ipywidgets.widgets.interaction.interactive
        The slider object after you have finished tweaking parameters
    sample_amount : float
        The percentage sampling variance (drawn from a uniform distribution) to vary intensities by. For example, a sample_amount of 0.05 will randomly sample +/- 5% above and below the chosen intensity for cells, media and device. Can be used to create a little bit of variance in the final training data. 
    randomise_hist_match : bool
        If true, histogram matching is randomly turned on and off each time a training sample is generated
    randomise_noise_match : bool
        If true, noise matching is randomly turned on and off each time a training sample is generated
    sim_length : int
        the length of the simulation which was run
    burn_in : int
        Number of frames to wait before generating training data. Can be used to ignore the start of the simulation where the trench only has 1 cell in it.
    n_samples : int
        The number of training images to generate
    save_dir : str
        The save directory of the training data
    
    """
    
    #media_multiplier, cell_multiplier, device_multiplier, sigma, scene_no, scale, match_histogram, match_noise, offset, debug_plot, noise_var = list(interactive_output.kwargs.values())
    media_multiplier, cell_multiplier, device_multiplier, sigma, scene_no, scale, match_fourier, match_histogram, match_noise, offset, debug_plot, noise_var, main_segments, scenes, kernel_params, resize_amount, real_image, image_params, error_params, x_border_expansion_coefficient,y_border_expansion_coefficient, fluorescence, defocus = list(interactive_output.kwargs.values())
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
        
    current_file_num = len(os.listdir(save_dir+"/convolutions"))
    #for z in range(n_samples):
    def generate_samples(z):
        _media_multiplier = np.random.uniform(1-sample_amount,1+sample_amount) * media_multiplier
        _cell_multiplier = np.random.uniform(1-sample_amount,1+sample_amount) * cell_multiplier
        _device_multiplier = np.random.uniform(1-sample_amount,1+sample_amount) * device_multiplier
        _sigma = np.random.uniform(1-sample_amount,1+sample_amount) * sigma
        _scene_no = np.random.randint(burn_in,sim_length-2)
        _noise_var = np.random.uniform(1-sample_amount,1+sample_amount) * noise_var
        if randomise_hist_match:
            _match_histogram = np.random.choice([True, False])
        else:
            _match_histogram = match_histogram
        if randomise_noise_match:
            _match_noise = np.random.choice([True, False])
        else:
            _match_noise = match_noise
        
        syn_image, mask = generate_test_comparison(_media_multiplier, _cell_multiplier, _device_multiplier, _sigma, _scene_no, scale, match_fourier, _match_histogram, match_noise, offset, debug_plot, noise_var, main_segments, scenes, kernel_params, resize_amount, real_image, image_params, error_params, x_border_expansion_coefficient,y_border_expansion_coefficient, fluorescence, defocus)
        
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
    Parallel(n_jobs=1)(delayed(generate_samples)(z) for z in tqdm(range(current_file_num,n_samples+current_file_num), desc="Sample generation"))
    
    
    
import numpy as np
from skimage.color import rgb2gray
from numpy import fft
from itertools import product
from PIL import Image
import copy
#from https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(phi, rho)
def pol2cart(phi,rho):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
def sfMatch(images,rescaling=0,tarmag=None):
    assert type(images) == type([]), 'The input must be a list.'

    numin = len(images)
    xs, ys = images[1].shape
    angs = np.zeros((xs,ys,numin))
    mags = np.zeros((xs,ys,numin))
    for x in range(numin):
        if len(images[x].shape) == 3:
            images[x] = rgb2gray(images[x])
        im1 = images[x]/255
        xs1, ys1 = im1.shape
        assert (xs == xs1) and (ys == ys1), 'All images must have the same size.'
        fftim1 = fft.fftshift(fft.fft2(im1))
        angs[:,:,x], mags[:,:,x] = cart2pol(np.real(fftim1),np.imag(fftim1))

    if tarmag is None:
        tarmag = np.mean(mags,2)

    xt, yt = tarmag.shape
    assert (xs == xt) and (ys == yt), 'The target spectrum must have the same size as the images.'
    f1 = np.linspace(-ys/2, ys/2-1,ys)
    f2 = np.linspace(-xs/2, xs/2-1,xs)
    XX, YY = np.meshgrid(f1,f2)
    t, r = cart2pol(XX,YY)
    if xs%2 == 1 or ys%2 == 1:
        r = np.round(r) - 1
    else:
        r = np.round(r)
    output_images = []
    for x in range(numin):
        fftim = mags[:,:,x]
        a = fftim.T.ravel()
        accmap = r.T.ravel()+1
        a2 = tarmag.T.ravel()
        en_old = np.array([np.sum([a[x] for x in y]) for y in [list(np.where(accmap==z)) for z in np.unique(accmap).tolist()]])
        en_new = np.array([np.sum([a2[x] for x in y]) for y in [list(np.where(accmap==z)) for z in np.unique(accmap).tolist()]])
        coefficient = en_new/en_old
        cmat = coefficient[(r).astype(int)]# coefficient[r+1]
        cmat[r>np.floor(np.max((xs,ys))/2)] = 0
        newmag = fftim*cmat
        XX, YY = pol2cart(angs[:,:,x],newmag)
        new = XX+YY*complex(0,1)
        output = np.real(fft.ifft2(fft.ifftshift(new)))
        if rescaling == 0:
            output = (output*255)
        output_images.append(output)
    if rescaling != 0:
        output_images = rescale_shine(output_images,rescaling)
    return output_images

def rescale_shine(images, option = 1):
    assert type(images) == type([]), 'The input must be a list.'
    assert option == 1 or option == 2, "Invalid rescaling option"
    numin = len(images)
    brightests = np.zeros((numin,1))
    darkests = np.zeros((numin,1))
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
            rescaled = (images[m] - the_darkest)/(the_brightest - the_darkest)*255
        elif option == 2:
            rescaled = (images[m] - avg_darkest)/(avg_brightest - avg_darkest)*255
        output_images.append(rescaled.astype(np.uint8))
    return output_images

def lumMatch(images, mask = None, lum = None):
    assert type(images) == type([]), 'The input must be a list.'
    assert (mask == None) or  type(mask) == type([]), 'The input mask must be a list.'
    

    numin = len(images)
    if (mask is None) and (lum is None):
        M = 0; S = 0
        for im in range(numin):
            if len(images[im].shape) == 3:
                images[im] = rgb2gray(images[im])
            M = M + np.mean(images[im])
            S = S + np.std(images[im])
        M = M/numin
        S = S/numin
        output_images = []
        for im in range(numin):
            im1 = copy.deepcopy(images[im])
            if np.std(im1) != 0:
                im1 = (im1 - np.mean(im1))/np.std(im1) * S + M
            else:
                im1[:,:] = M
            output_images.append(im1)
    elif (mask is None) and (lum != None):
        M = 0; S = 0
        for im in range(numin):
            if len(images[im].shape) == 3:
                images[im] = rgb2gray(images[im])
            M = lum[0]
            S = lum[1]
        M = M/numin
        S = S/numin
        output_images = []
        for im in range(numin):
            im1 = copy.deepcopy(images[im])
            if np.std(im1) != 0:
                im1 = (im1 - np.mean(im1))/np.std(im1) * S + M
            else:
                im1[:,:] = M
            output_images.append(im1)
    elif (mask != None) and (lum is None):
        M = 0; S = 0
        for im in range(numin):
            if len(images[im].shape) == 3:
                images[im] = rgb2gray(images[im])
            im1 = images[im]
            assert len(images) == len(mask), "The inputs must have the same length"
            m = mask[im]
            assert m.size == images[im].size, "Image and mask are not the same size"
            assert np.sum(m == 1) > 0, 'The mask must contain some ones.'
            M = M + np.mean(im1[m==1])
            S = S + np.mean(im1[m==1])
        M = M/numin
        S = S/numin
        output_images = []
        for im in range(numin):
            im1 = images[im]
            if type(mask) == type([]):
                m = mask[im] 
            if np.std(im1[m==1]):
                im1[m==1] = ( im1[m==1] - np.mean(im1[m==1]))/np.std(im1[m==1])* S + M
            else:
                im1[m==1] = M
            output_images.append(im1)
    elif (mask != None) and (lum != None):
        print("HI3")
        M = lum[0]; S = lum[1]
        output_images = []
        for im in range(numin):
            if len(images[im].shape) == 3:
                images[im] = rgb2gray(images[im])
            im1 = images[im]
            if len(mask) == 0:
                if np.std(im1) != 0.0:
                    im1 = (im1 - np.mean(im1))/np.std(im1) * S + M
                else:
                    im1[:,:] = M
            else:
                if type(mask) == type([]):
                    assert len(images) == len(mask), "The inputs must have the same length"
                    m = mask[im]
                assert m.size == images[im].size, "Image and mask are not the same size"
                assert np.sum(m == 1) > 0, 'The mask must contain some ones.'
                if np.std(im1[m==1]) != 0.0:
                    im1[m==1] = (im1[m==1] - np.mean(im1[m==1]))/np.std(im1[m==1])*S + M
                else:
                    im1[m==1] = M
            output_images.append(im1)
    return output_images