import pyglet
import pymunk
from pymunk.pyglet_util import DrawOptions
import sys
sys.path.insert(0,'/home/georgeos/Documents/GitHub/SYMPTOMM2')
from SYMPTOMM import cell_geometry
import matplotlib.pyplot as plt
import numpy as np
from SYMPTOMM.cell import Cell
import time

def create_space():
    return pymunk.Space()

def update_cell_lengths(cells):
    for cell in cells:
        cell.update_length()


def update_pm_cells(cells):
    for cell in cells:
        if cell.is_dividing() == True:
            daughter_details = cell.create_pm_cell()
            if len(daughter_details) > 2: # Really hacky. Needs fixing because sometimes this returns cell_body, cell shape. So this is a check to ensure that it's returing daughter_x, y and angle
                daughter = Cell(**daughter_details)
                cells.append(daughter)
        else:
            cell.create_pm_cell()

def update_cell_positions(cells):
    for cell in cells:
        cell.update_position()

def wipe_space(space):
    for body, poly in zip(space.bodies, space.shapes):
        if body.body_type == 0:
            space.remove(body)
            space.remove(poly)        


  
def step_and_update(dt, cells, space, phys_iters):


    new_cells = [cell for cell in cells if cell.body.position[1] > 500 or cell.body.position[1] < 0]
    cells = new_cells
    print(len(cells))
    for body, poly in zip(space.bodies, space.shapes):
        if body.body_type == 0:
            if body.position[1] > 500 or body.position[1] < 0:
                space.remove(body)
                space.remove(poly) 
    wipe_space(space)        
    update_cell_lengths(cells)
    update_pm_cells(cells)
    
    for _ in range(phys_iters):
        space.step(dt)
    update_cell_positions(cells)

def plot_scene(a, cells, savedir):
    if a%1 == 0:
        for cell in cells:
            vertices = cell.get_vertex_list()
            vertices = np.array(vertices)
            plt.plot(vertices[:,0], vertices[:,1])
            centroid = cell.position
            plt.scatter(centroid[0],centroid[1],s=100)
        plt.ylim(0,720)
        plt.xlim(0,720)
        plt.savefig(savedir+"/image_{}.png".format(str(a).zfill(3)))
        plt.clf()