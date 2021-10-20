import raster_geometry as rg
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
import pickle
import sys
#sys.path.insert(0,'/home/georgeos/Documents/GitHub/SYMPTOMM2')
import itertools
from joblib import Parallel, delayed
import pymunk 

def raster_cell(length, width):
    """Generates a 3D representation of a cell as a 2D image, where intensity is a measure of the depth of the cell at that point

    Parameters
    ----------
    length : float
        The length of the STRAIGHT part of the cell's wall. 
        Total length is cell_length + cell_width because the poles are models as semi-circles.
    cell_width : float
        Total thickness of the cell, defines the poles too.
    Returns
    -------
    array with cell thickness as intensities
    """
    radius = int(width/2)
    cyl_height = int(length - 2*radius)
    shape = 200 #200
    cylinder = rg.cylinder(
            shape = shape,
            height = cyl_height,
            radius = radius,
            axis=0,
            position=(0.5,0.5,0.5),
            smoothing=False)

    sphere1 = rg.sphere(shape,radius,((shape + cyl_height)/(2*shape),0.5,0.5))
    sphere2 = rg.sphere(shape,radius,((shape - cyl_height)/(2*shape),0.5,0.5))


    cell = (cylinder + sphere1 + sphere2)
    cell = cell[int(shape/2-cyl_height/2-radius-1):int(shape/2+cyl_height/2+radius+1),
                int(shape/2)-radius:int(shape/2)+radius,
               int(shape/2)-radius:int(shape/2)+radius]
    z,x,y = cell.nonzero()
    OPL_cell = np.sum(cell,axis=2)
    return OPL_cell

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
    return abs(np.sqrt((vertex1[0]-vertex2[0])**2 + (vertex1[1]-vertex2[1])**2))

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
        distance_ = get_distance(vertex_comb[0],vertex_comb[1])
        if distance_ > distance:
            distance = distance_
            farthest_vertices = vertex_comb
    return np.array(farthest_vertices)

def get_midpoint(vertex1, vertex2):
    """
    Get the midpoint between two vertices
    """
    x_mid = (vertex1[0]+vertex2[0])/2
    y_mid = (vertex1[1]+vertex2[1])/2
    return np.array([x_mid,y_mid])

def vertices_slope(vertex1, vertex2):
    """
    Get the slope between two vertices
    """
    return (vertex1[1] - vertex2[1])/(vertex1[0] - vertex2[0])

def midpoint_intercept(vertex1, vertex2):
    """
    Get the y-intercept of the line connecting two vertices
    """
    midpoint = get_midpoint(vertex1, vertex2)
    slope = vertices_slope(vertex1, vertex2)
    intercept = midpoint[1]-(slope*midpoint[0])
    return intercept

def get_centroid(vertices: list[tuple]) -> tuple:
    """Return the centroid of a list of vertices 
    
    Keyword arguments:
    vertices -- A list of tuples containing x,y coordinates.

    """
    return np.sum(vertices,axis=0)/len(vertices)

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
    OPL_cell = raster_cell(length = length, width=width)
    rotated_OPL_cell = rotate(OPL_cell,angle,resize=True,clip=False,preserve_range=True)
    cell_y, cell_x = (np.array(rotated_OPL_cell.shape)/2).astype(int)
    offset_y = rotated_OPL_cell.shape[0] - space[y-cell_y:y+cell_y,x-cell_x:x+cell_x].shape[0]
    offset_x = rotated_OPL_cell.shape[1] - space[y-cell_y:y+cell_y,x-cell_x:x+cell_x].shape[1]
    space[y-cell_y+100:y+cell_y+offset_y+100,x-cell_x+100:x+cell_x+offset_x+100] += rotated_OPL_cell

#deprecated
def gen_cell_props_for_draw_dep(cell_timeseries_lists):
    """
    
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
        angle = np.arctan(vertices_slope(farthest_vertices[0], farthest_vertices[1]))
        cell_properties.append([length, width, angle, centroid])
    return cell_properties

#deprecated
def draw_scene_dep(cell_properties):
    space_size = (800, 170)
    space = np.zeros(space_size)
    space_masks = np.zeros(space_size)
    offsets = 30
    for properties in cell_properties:
        length, width, angle, position = properties
        angle = np.rad2deg(angle) + 90
        x, y = np.array(position).astype(int) + offsets
        OPL_cell = raster_cell(length = length, width=width)
        rotated_OPL_cell = rotate(OPL_cell,angle,resize=True,clip=False,preserve_range=True)
        cell_y, cell_x = (np.array(rotated_OPL_cell.shape)/2).astype(int)
        offset_y = rotated_OPL_cell.shape[0] - space[y-cell_y:y+cell_y,x-cell_x:x+cell_x].shape[0]
        offset_x = rotated_OPL_cell.shape[1] - space[y-cell_y:y+cell_y,x-cell_x:x+cell_x].shape[1]
        space[y-cell_y:y+cell_y+offset_y,x-cell_x:x+cell_x+offset_x] += rotated_OPL_cell
        space_masks[y-cell_y:y+cell_y+offset_y,x-cell_x:x+cell_x+offset_x] += (rotated_OPL_cell > 7)
        space_masks = space_masks == 1
        space_masks = opening(space_masks,np.ones((1,8)))
    return space, space_masks

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
