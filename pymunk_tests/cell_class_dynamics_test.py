#%%
import pyglet
import pymunk
from pymunk.pyglet_util import DrawOptions
import sys
sys.path.insert(0,'/home/georgeos/Documents/GitHub/SYMPTOMM2')
from SYMPTOMM import cell_geometry
import matplotlib.pyplot as plt
import numpy as np
from cell_class_test import Cell
import time
space = pymunk.Space()

dt = 1/60
#%%
# Initialise space
cell1 = Cell(
    length = 40, 
    width = 20, 
    resolution = 20, 
    position = (200,320), 
    angle = 0.11, 
    space = space,
    dt = 1/60,
    growth_rate_constant = 1,
    max_length_mean = 80,
    max_length_var = 1
)

cell2 = Cell(
    length = 40,
    width = 20,
    resolution = 20,
    position = (250,300),
    angle = np.pi/4.2,
    space = space,
    dt = 1/60,
    growth_rate_constant = 1,
    max_length_mean = 80,
    max_length_var = 1
)
#%%


cells = [cell1, cell2]
def update_cell_lengths(cells):
    for cell in cells:
        cell.update_length()

def update_pm_cells(cells):
    for cell in cells:
        cell.create_pm_cell()

def update_cell_positions(cells):
    for cell in cells:
        cell.update_position()

def wipe_space(space):
    for body in space.bodies:
        space.remove(body)
    for poly in space.shapes:
        space.remove(poly)

def step_and_update(dt, cells, space, phys_iters):

    wipe_space(space)
    update_cell_lengths(cells)
    update_pm_cells(cells)
    for _ in range(phys_iters):
        space.step(dt)
    update_cell_positions(cells)





def plot_scene(a, cells):
    if a%3 == 0:
        for cell in cells:
            vertices = cell.get_vertex_list()
            vertices = np.array(vertices)
            plt.plot(vertices[:,0], vertices[:,1])
            centroid = cell.position
            plt.scatter(centroid[0],centroid[1],s=100)
        plt.ylim(0,720)
        plt.xlim(0,720)
        plt.savefig("/home/georgeos/Documents/GitHub/SYMPTOMM2/figures/{}.png".format(a))
        plt.clf()



# %%
for x in range(1000):
    plot_scene(x, cells)
    step_and_update(dt, cells, space, 50)
    print(cells[1].position)
# %%
