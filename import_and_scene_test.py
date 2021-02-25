import sys
sys.path.insert(0,'/home/georgeos/Documents/GitHub/SYMPTOMM2')
import pymunk
import matplotlib.pyplot as plt
import numpy as np
from SYMPTOMM import cell_geometry
from SYMPTOMM.cell import Cell
from SYMPTOMM.scene_functions import create_space, step_and_update
from SYMPTOMM.plotting import matplot_scene
from SYMPTOMM.trench_geometry import trench_creator
import pyglet
import pymunk
from pymunk.pyglet_util import DrawOptions
import pickle
window = pyglet.window.Window(180, 700, "MM Trench test", resizable=False)
options = DrawOptions()

space = create_space()
space.gravity = 0, -10
dt = 1/100

trench_creator(40,600,(0,0),space)
trench_creator(40,600,(90,0),space)

cell1 = Cell(
    length = 40, 
    width = 20, 
    resolution = 20, 
    position = (20,100), 
    angle = np.pi/3, 
    space = space,
    dt = 1/60,
    growth_rate_constant = 1,
    max_length_mean = 80,
    max_length_var = 0,
)

cell2 = Cell(
    length = 40, 
    width = 20, 
    resolution = 20, 
    position = (120,100), 
    angle = np.pi/3, 
    space = space,
    dt = 1/60,
    growth_rate_constant = 1,
    max_length_mean = 80,
    max_length_var = 0,
)
cells = [cell1,cell2]



@window.event
def on_draw():
    window.clear()
    space.debug_draw(options)

      

pyglet_draw = True

if pyglet_draw == True:
    if __name__ == "__main__":
        pyglet.clock.schedule_interval(step_and_update, dt, cells, space, 30)
        pyglet.app.run()
else:
    cell_timeseries = []
    for x in range(10000):
        step_and_update(dt=dt, cells=cells, space=space, phys_iters=30)
        cell_timeseries.append(cells)
    with open("output_pickles/cell_timeseries.p", "wb") as f:
        pickle.dump(cell_timeseries, f)
    with open("output_pickles/space.p", "wb") as f:
        pickle.dump(space, f)

