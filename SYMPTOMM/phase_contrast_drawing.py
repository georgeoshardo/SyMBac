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
import image_similarity_measures
from image_similarity_measures.quality_metrics import rmse, psnr, fsim, issm, sre, sam, uiq
from SYMPTOMM.general_drawing import *
from SYMPTOMM.phase_contrast_drawing import *

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



    
def generate_PC_OPL(main_segments, offset, scene, trench_multiplier,cell_multiplier,background_multiplier):

    def get_OPL_image():
        segment_1_top_left = (0 + offset, int(main_segments.iloc[0]["bb"][0] + offset))
        segment_1_bottom_right = (int(main_segments.iloc[0]["bb"][3] + offset), int(main_segments.iloc[0]["bb"][2] + offset))

        segment_2_top_left = (0 + offset, int(main_segments.iloc[1]["bb"][0] + offset))
        segment_2_bottom_right = (int(main_segments.iloc[1]["bb"][3] + offset), int(main_segments.iloc[1]["bb"][2] + offset))

        test_scene = np.zeros(scene.shape) + background_multiplier
        rr, cc = draw.rectangle(start = segment_1_top_left, end = segment_1_bottom_right, shape = test_scene.shape)
        test_scene[rr,cc] = 1 * trench_multiplier
        rr, cc = draw.rectangle(start = segment_2_top_left, end = segment_2_bottom_right, shape = test_scene.shape)
        test_scene[rr,cc] = 1 * trench_multiplier
        circ_midpoint_y = (segment_1_top_left[1] + segment_2_bottom_right[1])/2
        radius = (segment_1_top_left[1] - offset - (segment_2_bottom_right[1] - offset))/2
        circ_midpoint_x = (offset) + radius

        rr, cc = draw.rectangle(start = segment_2_top_left, end = (circ_midpoint_x,segment_1_top_left[1]), shape = test_scene.shape)
        test_scene[rr.astype(int),cc.astype(int)] = 1 * trench_multiplier
        rr, cc = draw.disk(center = (circ_midpoint_x, circ_midpoint_y), radius = radius, shape = test_scene.shape)
        rr_semi = rr[rr < (circ_midpoint_x + 1)]
        cc_semi = cc[rr < (circ_midpoint_x + 1)]
        test_scene[rr_semi,cc_semi] = background_multiplier
        no_cells = deepcopy(test_scene)
        test_scene += scene * cell_multiplier
        test_scene = test_scene[segment_2_top_left[0]:segment_1_bottom_right[0],segment_2_top_left[1]:segment_1_bottom_right[1]]
        no_cells = no_cells[segment_2_top_left[0]:segment_1_bottom_right[0],segment_2_top_left[1]:segment_1_bottom_right[1]]
        expanded_scene_no_cells = np.zeros((int(no_cells.shape[0]*1.2), no_cells.shape[1]*2)) + trench_multiplier
        expanded_scene_no_cells[expanded_scene_no_cells.shape[0] - no_cells.shape[0]:,int(no_cells.shape[1]/2):int(no_cells.shape[1]/2) + no_cells.shape[1]] = no_cells

        expanded_scene = np.zeros((int(test_scene.shape[0]*1.2), test_scene.shape[1]*2)) + trench_multiplier
        expanded_scene[expanded_scene.shape[0] - test_scene.shape[0]:,int(test_scene.shape[1]/2):int(test_scene.shape[1]/2) + test_scene.shape[1]] = test_scene
        return expanded_scene, expanded_scene_no_cells
    expanded_scene, expanded_scene_no_cells = get_OPL_image()
    if expanded_scene is None:
        main_segments = main_segments.reindex(index=main_segments.index[::-1])
        expanded_scene, expanded_scene_no_cells = get_OPL_image()
    return expanded_scene, expanded_scene_no_cells


def gaussian_2D(size, σ):
    x = np.linspace(0,size,size)
    μ = np.mean(x)
    A = 1/(σ*np.sqrt(2*np.pi))
    B = np.exp(-1/2 * (x-μ)**2/(σ**2))
    _gaussian_1D = A*B
    _gaussian_2D = np.outer(_gaussian_1D,_gaussian_1D)
    return _gaussian_2D


## Maybe move to PSF file?
def somb(x):
    z = np.zeros(x.shape)
    x = np.abs(x)
    idx = np.nonzero(x)
    z[idx] = 2*jv(1,np.pi*x[idx])/(np.pi*x[idx])
    return z

def get_phase_contrast_kernel(R,W,radius,scale,F,sigma,λ):
    scale1 = 1000 # micron per millimeter
    F = F * scale1 # to microm
    Lambda = λ # in micron % wavelength of light
    R = R * scale1 # to microm
    W = W * scale1 # to microm
    #The corresponding point spread kernel function for the negative phase contrast 

    meshgrid_arrange = np.arange(-radius,radius + 1,1)
    [xx,yy] = np.meshgrid(meshgrid_arrange,meshgrid_arrange)
    rr = np.sqrt(xx**2 + yy**2)*scale
    rr_dl = rr*(1/F)*(1/Lambda); # scaling with F and Lambda for dimension correction
    kernel1 = np.pi*R**2*somb(2*R*rr_dl);     
    kernel2 = np.pi*(R-W)**2*somb(2*(R-W)*rr_dl)


    kernel = kernel1 - kernel2
    kernel = kernel/np.max(kernel)
    kernel[radius,radius] = kernel[radius,radius] + 1
    kernel = -kernel/np.sum(kernel)
    gaussian = gaussian_2D(radius*2+1, sigma)
    kernel = kernel * gaussian
    return kernel

def get_condensers():
    condensers = {
    "Ph1": (0.45, 3.75, 24),
    "Ph2": (0.8, 5.0, 24),
    "Ph3": (1.0, 9.5, 24),
    "Ph4": (1.5, 14.0, 24),
    "PhF": (1.5, 19.0, 25)
    } #W, R, Diameter
    return condensers
    
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

    σ, trench_multiplier, cell_multiplier, background_multiplier, trench_length, trench_width, cell_max_length, cell_width = z
    print(z)

    cell_timeseries, space = run_simulation(trench_length, trench_width, cell_max_length, cell_width)
    main_segments = get_trench_segments(space)
    ID_props = generate_curve_props(cell_timeseries)
    cell_timeseries_properties = Parallel(n_jobs=14)(
        delayed(gen_cell_props_for_draw)(a, ID_props) for a in cell_timeseries[-5:-1])
    do_transformation = True
    scenes = Parallel(n_jobs=14)(
        delayed(draw_scene)(cell_properties, do_transformation) for cell_properties in tqdm(cell_timeseries_properties))

    # expanded_scene = generate_PC_OPL(trench_multiplier,cell_multiplier,background_multiplier)
    OPL_scenes = [
        generate_PC_OPL(main_segments, offset, scene[1], trench_multiplier, cell_multiplier, background_multiplier) for
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