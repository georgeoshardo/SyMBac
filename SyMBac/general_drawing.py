import raster_geometry as rg
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
import pickle
import sys
from skimage.transform import rescale, resize, downscale_local_mean
#sys.path.insert(0,'/home/georgeos/Documents/GitHub/SYMPTOMM2')
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
from skimage.exposure import rescale_intensity
import importlib, warnings
if importlib.util.find_spec("cupy") is None:
    from scipy.signal import convolve2d as cuconvolve
    warnings.warn("Could not load CuPy for SyMBac, are you using a GPU? Defaulting to CPU convolution.")
    def convolve_rescale(image,kernel,rescale_factor, rescale_int):
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

        output = cuconvolve(image,kernel,mode="same")
        #output = output.get()
        output = rescale(output, rescale_factor, anti_aliasing=False)

        if rescale_int:
            output = rescale_intensity(output.astype(np.float32), out_range=(0,1))
        return output
else:
    import cupy as cp
    from cupyx.scipy.ndimage import convolve as cuconvolve
    def convolve_rescale(image,kernel,rescale_factor, rescale_int):
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

        output = cuconvolve(cp.array(image),cp.array(kernel))
        output = output.get()
        output = rescale(output, rescale_factor, anti_aliasing=False)

        if rescale_int:
            output = rescale_intensity(output.astype(np.float32), out_range=(0,1))
        return output

def generate_curve_props(cell_timeseries):
    """
    Generates individual cell curvature properties. 3 parameters for each cell, which are passed to a cosine function to modulate the cell's curvature. 
    
    Paramters
    ---------
    cell_timeseries : list
        The output of run_simulation()
    
    Returns
    -------
    A numpy array of unique curvature properties for each cell in the simulation
    """
    
    #Get unique cell IDs
    IDs = []
    for cell_list in cell_timeseries:
        for cell in cell_list:
            IDs.append(cell.ID)
    IDs = np.array(IDs)
    unique_IDs = np.unique(IDs)
    #For each cell, assign random curvature properties
    ID_props = []
    for ID in unique_IDs:
        freq_modif = (np.random.uniform(0.9,1.1)) # Choose one per cell
        amp_modif = (np.random.uniform(0.9,1.1)) # Choose one per cell
        phase_modif = np.random.uniform(-1,1)  # Choose one per cell
        ID_props.append([int(ID), freq_modif, amp_modif, phase_modif])
    ID_propps = np.array(ID_props)
    ID_propps[:,0] = ID_propps[:,0].astype(int)
    return np.array(ID_props)

def gen_cell_props_for_draw(cell_timeseries_lists, ID_props):
    """
    Parameters
    ----------
    cell_timeseries_list : list
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
            x,y = v.rotated(shape.body.angle) + shape.body.position #.rotated(self.shape.body.angle)
            vertices.append((x,y))
        vertices = np.array(vertices)

        centroid = get_centroid(vertices) 
        farthest_vertices = find_farthest_vertices(vertices)
        length = get_distance(farthest_vertices[0],farthest_vertices[1])
        width = cell.width
        #angle = np.arctan(vertices_slope(farthest_vertices[0], farthest_vertices[1]))
        angle = np.arctan2((farthest_vertices[0] - farthest_vertices[1])[1],(farthest_vertices[0] - farthest_vertices[1])[0])
        angle = np.rad2deg(angle)+90

        ID, freq_modif, amp_modif, phase_modif = ID_props[ID_props[:,0] == cell.ID][0]
        phase_mult = 20
        cell_properties.append([length, width, angle, centroid, freq_modif, amp_modif, phase_modif,phase_mult])
    return cell_properties


def raster_cell(length, width):
    L = int(np.rint(length))
    W = int(np.rint(width))
    new_cell = np.zeros((L,W))
    R = (W-1)/2


    x_cyl = np.arange(0,2*R+1,1)
    I_cyl = np.sqrt(R**2 - (x_cyl-R)**2)
    L_cyl = L - W
    new_cell[int(W/2):-int(W/2),:] = I_cyl

    x_sphere = np.arange(0,int(W/2),1)
    sphere_Rs = np.sqrt((R)**2 - (x_sphere-R)**2)
    sphere_Rs = np.rint(sphere_Rs).astype(int)

    for c in range(len(sphere_Rs)):
        R_ = sphere_Rs[c]
        x_cyl = np.arange(0,R_,1)
        I_cyl = np.sqrt(R_**2 - (x_cyl-R_)**2)
        new_cell[c,int(W/2)-sphere_Rs[c]:int(W/2)+sphere_Rs[c]] = np.concatenate((I_cyl,I_cyl[::-1]))
        new_cell[L-c-1,int(W/2)-sphere_Rs[c]:int(W/2)+sphere_Rs[c]] = np.concatenate((I_cyl,I_cyl[::-1]))
    new_cell = new_cell.astype(int)
    return new_cell


def get_distance(vertex1, vertex2):
    return abs(np.sqrt((vertex1[0]-vertex2[0])**2 + (vertex1[1]-vertex2[1])**2))

def find_farthest_vertices(vertex_list):
    vertex_combs = list(itertools.combinations(vertex_list, 2))
    distance = 0
    farthest_vertices = 0
    for vertex_comb in vertex_combs:
        distance_ = get_distance(vertex_comb[0],vertex_comb[1])
        if distance_ > distance:
            distance = distance_
            farthest_vertices = vertex_comb
    return np.array(farthest_vertices)

def get_midpoint(vertex1, vertex2):
    x_mid = (vertex1[0]+vertex2[0])/2
    y_mid = (vertex1[1]+vertex2[1])/2
    return np.array([x_mid,y_mid])

def vertices_slope(vertex1, vertex2):
    return (vertex1[1] - vertex2[1])/(vertex1[0] - vertex2[0])

def midpoint_intercept(vertex1, vertex2):
    midpoint = get_midpoint(vertex1, vertex2)
    slope = vertices_slope(vertex1, vertex2)
    intercept = midpoint[1]-(slope*midpoint[0])
    return intercept

def get_centroid(vertices):
    """Return the centroid of a list of vertices 
    
    Keyword arguments:
    vertices -- A list of tuples containing x,y coordinates.

    """
    return np.sum(vertices,axis=0)/len(vertices)

def place_cell(length, width, angle, position, space):
    angle = np.rad2deg(angle)
    x, y = np.array(position).astype(int)
    OPL_cell = raster_cell(length = length, width=width)
    rotated_OPL_cell = rotate(OPL_cell,angle,resize=True,clip=False,preserve_range=True)
    cell_y, cell_x = (np.array(rotated_OPL_cell.shape)/2).astype(int)
    offset_y = rotated_OPL_cell.shape[0] - space[y-cell_y:y+cell_y,x-cell_x:x+cell_x].shape[0]
    offset_x = rotated_OPL_cell.shape[1] - space[y-cell_y:y+cell_y,x-cell_x:x+cell_x].shape[1]
    space[y-cell_y+100:y+cell_y+offset_y+100,x-cell_x+100:x+cell_x+offset_x+100] += rotated_OPL_cell
    
def transform_func(amp_modif, freq_modif, phase_modif):
    def perm_transform_func(x, amp_mult, freq_mult, phase_mult):
        return (amp_mult*amp_modif*np.cos((x/(freq_mult * freq_modif) - phase_mult * phase_modif)*np.pi)).astype(int)
    return perm_transform_func



def scene_plotter(scene_array,output_dir,name,a,matplotlib_draw):
    if matplotlib_draw == True:
        plt.figure(figsize=(3,10))
        plt.imshow(scene_array)
        plt.tight_layout()
        plt.savefig(output_dir+"/{}_{}.png".format(name,str(a).zfill(3)))
        plt.clf()
        plt.close('all')
    else:
        im = Image.fromarray(scene_array.astype(np.uint8))
        im.save(output_dir+"/{}_{}.tif".format(name,str(a).zfill(3)))
        

def make_images_same_shape(real_image,synthetic_image, rescale_int = True):
    """ Makes a synthetic image the same shape as the real image"""
    
    
    assert real_image.shape[0] < synthetic_image.shape[0], "Real image has a higher diemsion on axis 0, increase y_border_expansion_coefficient"
    assert real_image.shape[1] < synthetic_image.shape[1], "Real image has a higher diemsion on axis 1, increase x_border_expansion_coefficient"

    x_diff = synthetic_image.shape[1] - real_image.shape[1]
    remove_from_left, remove_from_right = div_odd(x_diff)
    y_diff = synthetic_image.shape[0] - real_image.shape[0]
    if real_image.shape[1]%2 == 0:
        if synthetic_image.shape[1]%2 == 0:
            if y_diff > 0:
                synthetic_image = synthetic_image[y_diff:,remove_from_left-1:-remove_from_right]
            else:
                synthetic_image = synthetic_image[:,remove_from_left:-remove_from_right]
                real_image = real_image[abs(y_diff):,:]
        elif synthetic_image.shape[1]%2 == 1:
            if y_diff > 0:
                synthetic_image = synthetic_image[y_diff:,remove_from_left:-remove_from_right]
            else:
                synthetic_image = synthetic_image[:,remove_from_left:-remove_from_right]
                real_image = real_image[abs(y_diff):,:]
    elif real_image.shape[1]%2 == 1:
        if synthetic_image.shape[1]%2 == 0:
            if y_diff > 0:
                synthetic_image = synthetic_image[y_diff:,remove_from_left:-remove_from_right]
            else:
                synthetic_image = synthetic_image[:,remove_from_left:-remove_from_right]
                real_image = real_image[abs(y_diff):,:]
        elif synthetic_image.shape[1]%2 == 1:
            if y_diff > 0:
                synthetic_image = synthetic_image[y_diff:,remove_from_left-1:-remove_from_right]
            else:
                synthetic_image = synthetic_image[:,remove_from_left:-remove_from_right]
                real_image = real_image[abs(y_diff):,:]

    if rescale_int:
        real_image = rescale_intensity(real_image.astype(np.float32), out_range=(0,1))
        synthetic_image = rescale_intensity(synthetic_image.astype(np.float32), out_range=(0,1))
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
    return (int(1.2*max_y), int(1.5*max_x))

div_odd = lambda n: (n//2, n//2 + 1)
perc_diff = lambda a, b: abs(a-b)/((a+b)/2)



#def draw_scene(cell_properties, do_transformation, mask_threshold, space_size, x_offset, y_offset, label_masks):
#    space_size = np.array(space_size) # 1000, 200 a good value
#    space = np.zeros(space_size)
#    space_masks = np.zeros(space_size)
#    offsets = offset
#    if label_masks:
#        colour_label = 1
#    for properties in cell_properties:
#        length, width, angle, position, freq_modif, amp_modif, phase_modif,phase_mult = properties
#        length = length; width = width ; position = np.array(position) 
#        angle = np.rad2deg(angle) - 90
#        x = np.array(position).astype(int)[0] + x_offset
#        y = np.array(position).astype(int)[1] + y_offset
#        OPL_cell = raster_cell(length = length, width=width)

#        if do_transformation:
#            OPL_cell_2 = np.zeros((OPL_cell.shape[0],int(OPL_cell.shape[1]*2)))
#            midpoint = int(np.median(range(OPL_cell_2.shape[1])))
#            OPL_cell_2[:,midpoint-int(OPL_cell.shape[1]/2):midpoint-int(OPL_cell.shape[1]/2)+OPL_cell.shape[1]] = OPL_cell
#            roll_coords = np.array(range(OPL_cell_2.shape[0]))
#            freq_mult = (OPL_cell_2.shape[0])
#            amp_mult = OPL_cell_2.shape[1]/10
#            sin_transform_cell = transform_func(amp_modif, freq_modif, phase_modif)
#            roll_amounts = sin_transform_cell(roll_coords,amp_mult,freq_mult,phase_mult)
#            for B in roll_coords:
#                OPL_cell_2[B,:] = np.roll(OPL_cell_2[B,:], roll_amounts[B])
#            OPL_cell = (OPL_cell_2)

#        rotated_OPL_cell = rotate(OPL_cell,angle,resize=True,clip=False,preserve_range=True)
#        cell_y, cell_x = (np.array(rotated_OPL_cell.shape)/2).astype(int)
#        offset_y = rotated_OPL_cell.shape[0] - space[y-cell_y:y+cell_y,x-cell_x:x+cell_x].shape[0]
#        offset_x = rotated_OPL_cell.shape[1] - space[y-cell_y:y+cell_y,x-cell_x:x+cell_x].shape[1]
#        assert y > cell_y, "Cell has {} negative pixels in y coordinate, try increasing your offset".format(y - cell_y)
#        assert x > cell_x, "Cell has negative pixels in x coordinate, try increasing your offset"
#        space[
#            y-cell_y:y+cell_y+offset_y,
#            x-cell_x:x+cell_x+offset_x
#        ] += rotated_OPL_cell
#        if label_masks:
#            space_masks[y-cell_y:y+cell_y+offset_y,x-cell_x:x+cell_x+offset_x] = (rotated_OPL_cell > 0)*colour_label
#            colour_label += 1
#        else:
#            space_masks[y-cell_y:y+cell_y+offset_y,x-cell_x:x+cell_x+offset_x] += (rotated_OPL_cell > mask_threshold)*colour_label
#            space_masks = space_masks == 1


        #space_masks = opening(space_masks,np.ones((2,11)))
#    return space, space_masks

#def raster_cell(length, width):
 #   #TODO: make this FASTER
 #   radius = int(width/2)
 #   cyl_height = int(length - 2*radius)
 #   shape = 500 #200
 #   cylinder = rg.cylinder(
 #           shape = shape,
 #           height = cyl_height,
 #           radius = radius,
 #           axis=0,
 #           position=(0.5,0.5,0.5),
 #           smoothing=False)##

 #   sphere1 = rg.sphere(shape,radius,((shape + cyl_height)/(2*shape),0.5,0.5))
 #   sphere2 = rg.sphere(shape,radius,((shape - cyl_height)/(2*shape),0.5,0.5))#


 #   cell = (cylinder + sphere1 + sphere2)
 #   cell = cell[int(shape/2-cyl_height/2-radius-1):int(shape/2+cyl_height/2+radius+1),
 #               int(shape/2)-radius:int(shape/2)+radius,
 #              int(shape/2)-radius:int(shape/2)+radius]
 #   z,x,y = cell.nonzero()
 #   OPL_cell = np.sum(cell,axis=2)
 #   return OPL_cell
