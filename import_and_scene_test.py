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
from copy import deepcopy
window = pyglet.window.Window(700, 700, "MM Trench test", resizable=False)
options = DrawOptions()

space = create_space()
space.gravity = 0, 0
dt = 1/100


trench_length = 600

trench_creator(25,trench_length,(0,0),space) # Coordinates of bottom left corner of the trench

cell1 = Cell(
    length = 40, 
    width = 20, 
    resolution = 60, 
    position = (30,40), 
    angle = np.pi/2, 
    space = space,
    dt = 1/60,
    growth_rate_constant = 1,
    max_length_mean = 80,
    max_length_var = 0
)

cells = [cell1]

#body = pymunk.Body(1,1666,pymunk.Body.KINEMATIC)
#body.position = 120,100
#poly = pymunk.Poly.create_box(body,size=(40,40))
#space.add(body,poly)

@window.event
def on_draw():
    window.clear()
    space.debug_draw(options)

      

pyglet_draw = False
matplot_draw = False

phys_iters = 75

if pyglet_draw == True:
    if __name__ == "__main__":
        pyglet.clock.schedule_interval(step_and_update, dt, cells, space, phys_iters,ylim=trench_length)
        pyglet.app.run()
elif matplot_draw:
    for x in range(700):
        step_and_update(dt=dt, cells=cells, space=space, phys_iters=phys_iters,ylim=trench_length)
        matplot_scene(x,cells, "/home/georgeos/Documents/GitHub/SYMPTOMM2/figures")
else:
    cell_timeseries = []
    for x in range(400):
        cells = step_and_update(dt=dt, cells=cells, space=space, phys_iters=phys_iters,ylim=trench_length)
        cell_timeseries.append(deepcopy(cells))
    with open("output_pickles/cell_timeseries.p", "wb") as f:
        pickle.dump(cell_timeseries, f)
    with open("output_pickles/space.p", "wb") as f:
        pickle.dump(space, f)


