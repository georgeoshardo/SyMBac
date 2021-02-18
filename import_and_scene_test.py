sys.path.insert(0,'/home/georgeos/Documents/GitHub/SYMPTOMM2')
import pymunk
import sys
from SYMPTOMM import cell_geometry
import matplotlib.pyplot as plt
import numpy as np
from SYMPTOMM.cell import Cell
from SYMPTOMM.scene_functions import step_and_update
from SYMPTOMM.plotting import plot_scene

space = pymunk.Space()
cell1 = Cell(
    length = 40, 
    width = 20, 
    resolution = 20, 
    position = (720/2,720/2), 
    angle = np.pi/3, 
    space = space,
    dt = 1/60,
    growth_rate_constant = 1,
    max_length_mean = 80,
    max_length_var = 0,
)

cells = [cell1]


savedir = "/home/georgeos/Documents/GitHub/SYMPTOMM2/figures"
dt = 1/60
for x in range(400):
    plot_scene(x, cells, savedir)
    step_and_update(dt, cells, space, 5)