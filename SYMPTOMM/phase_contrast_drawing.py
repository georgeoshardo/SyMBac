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
sys.path.insert(0,'/home/georgeos/Documents/GitHub/SYMPTOMM2')
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
from SYMPTOMM import PSF
from matplotlib_scalebar.scalebar import ScaleBar
from cupyx.scipy.ndimage import convolve as cuconvolve
import tifffile
from skimage.exposure import match_histograms
import cupy as cp
from scipy.optimize import dual_annealing, shgo
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import basinhopping
#import image_similarity_measures
#from image_similarity_measures.quality_metrics import rmse, psnr, fsim, issm, sre, sam, uiq
from SYMPTOMM.general_drawing import *
from SYMPTOMM.phase_contrast_drawing import *
from SYMPTOMM.scene_functions import *
from SYMPTOMM.trench_geometry import *
from SYMPTOMM.PSF import *
import os
import skimage
##From here, prototyping phase contrast
def get_trench_segments(space):
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
    
def run_simulation(trench_length, trench_width, cell_max_length, cell_width, sim_length, pix_mic_conv, gravity, phys_iters, max_length_var, width_var):
    
    
    space = create_space()
    space.gravity = 0, gravity # arbitrary units, negative is toward trench pole
    dt = 1/100 #time-step per frame
    pix_mic_conv = 1/pix_mic_conv # micron per pixel
    scale_factor = pix_mic_conv * 3 # resolution scaling factor 

    trench_length = trench_length*scale_factor
    trench_width = trench_width*scale_factor
    trench_creator(trench_width,trench_length,(35,0),space) # Coordinates of bottom left corner of the trench
    #trench_creator(35,trench_length,(35*3,0),space) # Coordinates of bottom left corner of the trench
    #trench_creator(35,trench_length,(35*5,0),space) # Coordinates of bottom left corner of the trench

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


    cells = [cell1]
    cell_timeseries = []
    phys_iters = phys_iters
    for x in tqdm(range(sim_length+250),desc="Simulation Progress"):
        cells = step_and_update(dt=dt, cells=cells, space=space, phys_iters=phys_iters,ylim=trench_length)
        if x > 250:
            cell_timeseries.append(deepcopy(cells))
    return cell_timeseries, space

def get_similarity_metrics(real_image,synthetic_image):
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

def generate_PC_OPL(main_segments, offset, scene, mask, media_multiplier,cell_multiplier,device_multiplier, y_border_expansion_coefficient, x_border_expansion_coefficient):
    def get_OPL_image(mask):
        segment_1_top_left = (0 + offset, int(main_segments.iloc[0]["bb"][0] + offset))
        segment_1_bottom_right = (int(main_segments.iloc[0]["bb"][3] + offset), int(main_segments.iloc[0]["bb"][2] + offset))

        segment_2_top_left = (0 + offset, int(main_segments.iloc[1]["bb"][0] + offset))
        segment_2_bottom_right = (int(main_segments.iloc[1]["bb"][3] + offset), int(main_segments.iloc[1]["bb"][2] + offset))

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
        test_scene = np.where(no_cells != media_multiplier, test_scene, media_multiplier)
        test_scene = test_scene[segment_2_top_left[0]:segment_1_bottom_right[0],segment_2_top_left[1]:segment_1_bottom_right[1]]
        
        mask = np.where(no_cells != media_multiplier, mask, 0)
        mask_resized = mask[segment_2_top_left[0]:segment_1_bottom_right[0],segment_2_top_left[1]:segment_1_bottom_right[1]]
        
        no_cells = no_cells[segment_2_top_left[0]:segment_1_bottom_right[0],segment_2_top_left[1]:segment_1_bottom_right[1]]
        expanded_scene_no_cells = np.zeros((int(no_cells.shape[0]*y_border_expansion_coefficient), int(no_cells.shape[1]*x_border_expansion_coefficient))) + media_multiplier
        expanded_scene_no_cells[expanded_scene_no_cells.shape[0] - no_cells.shape[0]:,
                                int(expanded_scene_no_cells.shape[1]/2-int(test_scene.shape[1]/2)):int(expanded_scene_no_cells.shape[1]/2-int(test_scene.shape[1]/2)) + no_cells.shape[1]] = no_cells

        expanded_scene = np.zeros((int(test_scene.shape[0]*y_border_expansion_coefficient), int(test_scene.shape[1]*x_border_expansion_coefficient))) + media_multiplier
        expanded_scene[expanded_scene.shape[0] - test_scene.shape[0]:,
                       int(expanded_scene.shape[1]/2-int(test_scene.shape[1]/2)):int(expanded_scene.shape[1]/2-int(test_scene.shape[1]/2)) + test_scene.shape[1]] = test_scene
        
        expanded_mask = np.zeros((int(test_scene.shape[0]*y_border_expansion_coefficient), int(test_scene.shape[1]*x_border_expansion_coefficient)))
        expanded_mask[expanded_mask.shape[0] - test_scene.shape[0]:,
                      int(expanded_mask.shape[1]/2-int(test_scene.shape[1]/2)):int(expanded_mask.shape[1]/2-int(test_scene.shape[1]/2)) + test_scene.shape[1]] = mask_resized
        
        return expanded_scene, expanded_scene_no_cells, expanded_mask
    expanded_scene, expanded_scene_no_cells, expanded_mask = get_OPL_image(mask)
    if expanded_scene is None:
        main_segments = main_segments.reindex(index=main_segments.index[::-1])
        expanded_scene, expanded_scene_no_cells, expanded_mask = get_OPL_image(mask)
    return expanded_scene, expanded_scene_no_cells, expanded_mask


def generate_test_comparison(media_multiplier, cell_multiplier, device_multiplier, sigma, scene_no, scale, match_histogram, match_noise, offset, debug_plot, noise_var, main_segments, scenes, kernel_params, resize_amount, real_image, image_params, error_params, x_border_expansion_coefficient,y_border_expansion_coefficient):
    
    
    expanded_scene, expanded_scene_no_cells, expanded_mask = generate_PC_OPL(
        main_segments=main_segments,
        offset=offset,
        scene = scenes[scene_no][0],
        mask = scenes[scene_no][1],
        media_multiplier=media_multiplier,
        cell_multiplier=cell_multiplier,
        device_multiplier=device_multiplier,
        x_border_expansion_coefficient = x_border_expansion_coefficient,
        y_border_expansion_coefficient = y_border_expansion_coefficient
    )



    #R,W,radius,scale,F,_,λ = kernel_params
    R,W,radius,scale,NA,n,_,λ = kernel_params
    
    
    real_media_mean, real_cell_mean, real_device_mean, real_means, real_media_var, real_cell_var, real_device_var, real_vars = image_params
    mean_error,media_error,cell_error,device_error,mean_var_error,media_var_error,cell_var_error,device_var_error = error_params
    
    
    kernel = get_phase_contrast_kernel(R=R, W=W, radius=radius, scale=scale, NA=NA,n=n, sigma=sigma, λ=λ)
    #kernel = get_phase_contrast_kernel(R=R, W=W, radius=radius, scale=scale, F=F, sigma=sigma, λ=λ) # 



    convolved = convolve_rescale(expanded_scene, kernel, 1/resize_amount, rescale_int = True)
    real_resize, expanded_resized = make_images_same_shape(real_image,convolved, rescale_int=True)
    
    
    if match_histogram:
        matched = match_histograms(expanded_resized, real_resize, multichannel=False)
    else:
        matched = expanded_resized
    
    
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
    
    expanded_device_mask = expanded_scene_no_cells == media_multiplier
    expanded_device_mask = rescale(expanded_device_mask, 1/resize_amount, anti_aliasing=False)
    real_resize, expanded_device_mask = make_images_same_shape(real_image,expanded_device_mask, rescale_int=True)
    just_device = expanded_device_mask * noisy_img
    
    
    
    
    simulated_means = np.array([just_media[np.where(just_media)].mean(), just_cells[np.where(just_cells)].mean(), just_device[np.where(just_device)].mean()])
    simulated_vars = np.array([just_media[np.where(just_media)].var(), just_cells[np.where(just_cells)].var(), just_device[np.where(just_device)].var()])

    
    
    
    
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
    space_size = np.array(space_size) # 1000, 200 a good value
    space = np.zeros(space_size)
    space_masks = np.zeros(space_size)
    if label_masks:
        colour_label = 1
    for properties in cell_properties:
        length, width, angle, position, freq_modif, amp_modif, phase_modif,phase_mult = properties
        length = length; width = width ; position = np.array(position) 
        angle = np.rad2deg(angle) - 90
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

        rotated_OPL_cell = rotate(OPL_cell,angle,resize=True,clip=False,preserve_range=True)
        cell_y, cell_x = (np.array(rotated_OPL_cell.shape)/2).astype(int)
        offset_y = rotated_OPL_cell.shape[0] - space[y-cell_y:y+cell_y,x-cell_x:x+cell_x].shape[0]
        offset_x = rotated_OPL_cell.shape[1] - space[y-cell_y:y+cell_y,x-cell_x:x+cell_x].shape[1]
        assert y > cell_y, "Cell has {} negative pixels in y coordinate, try increasing your offset".format(y - cell_y)
        assert x > cell_x, "Cell has negative pixels in x coordinate, try increasing your offset"
        space[
            y-cell_y:y+cell_y+offset_y,
            x-cell_x:x+cell_x+offset_x
        ] += rotated_OPL_cell
        if label_masks:
            space_masks[y-cell_y:y+cell_y+offset_y,x-cell_x:x+cell_x+offset_x] = (rotated_OPL_cell > 0)*colour_label
            colour_label += 1
        else:
            space_masks[y-cell_y:y+cell_y+offset_y,x-cell_x:x+cell_x+offset_x] += (rotated_OPL_cell > mask_threshold)*colour_label
            space_masks = space_masks == 1


        #space_masks = opening(space_masks,np.ones((2,11)))
    return space, space_masks

def generate_training_data(interactive_output, sample_amount, randomise_hist_match, randomise_noise_match, sim_length, burn_in, n_samples, save_dir):
    #media_multiplier, cell_multiplier, device_multiplier, sigma, scene_no, scale, match_histogram, match_noise, offset, debug_plot, noise_var = list(interactive_output.kwargs.values())
    media_multiplier, cell_multiplier, device_multiplier, sigma, scene_no, scale, match_histogram, match_noise, offset, debug_plot, noise_var, main_segments, scenes, kernel_params, resize_amount, real_image, image_params, error_params, x_border_expansion_coefficient,y_border_expansion_coefficient = list(interactive_output.kwargs.values())
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
        
        syn_image, mask = generate_test_comparison(_media_multiplier, _cell_multiplier, _device_multiplier, _sigma, _scene_no, scale, _match_histogram, match_noise, offset, debug_plot, noise_var, main_segments, scenes, kernel_params, resize_amount, real_image, image_params, error_params, x_border_expansion_coefficient,y_border_expansion_coefficient)
        
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