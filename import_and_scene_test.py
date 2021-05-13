import sys
sys.path.insert(0,'/home/georgeos/Documents/GitHub/SYMPTOMM2')
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
window = pyglet.window.Window(700, 1100, "MM Trench test", resizable=False)
options = DrawOptions()

space = create_space()
space.gravity = 0, -0.3
dt = 1/100


pix_mic_conv = 10
scale_factor = 3 * pix_mic_conv

trench_length = 17*scale_factor

trench_creator(1.1*scale_factor,trench_length,(35,0),space) # Coordinates of bottom left corner of the trench
trench_creator(35,trench_length,(35*3,0),space) # Coordinates of bottom left corner of the trench
trench_creator(35,trench_length,(35*5,0),space) # Coordinates of bottom left corner of the trench

cell1 = Cell(
    length = 3.5*scale_factor,  
    width = 0.95*scale_factor, 
    resolution = 60, 
    position = (20+35,40), 
    angle = 0.8, 
    space = space,
    dt = 1/60,
    growth_rate_constant = 1,
    max_length = 3.5*scale_factor,
    max_length_mean =5*scale_factor,
    max_length_var = 0.1*np.sqrt(scale_factor),
    width_var = 0.03*np.sqrt(scale_factor),
    width_mean = 0.75*scale_factor
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

      

pyglet_draw = True
matplot_draw = False

phys_iters = 200

if pyglet_draw == True:
    if __name__ == "__main__":
        pyglet.clock.schedule_interval(step_and_update, dt, cells, space, phys_iters,ylim=trench_length)
        pyglet.app.run()
elif matplot_draw:
    for x in range(200):
        step_and_update(dt=dt, cells=cells, space=space, phys_iters=phys_iters,ylim=trench_length)
        matplot_scene(x,cells, "/home/georgeos/Documents/GitHub/SYMPTOMM2/figures")
else:
    cell_timeseries = []
    for x in range(650):
        cells = step_and_update(dt=dt, cells=cells, space=space, phys_iters=phys_iters,ylim=trench_length)
        if x > 250:
            cell_timeseries.append(deepcopy(cells))
    with open("/home/georgeos/Storage/Dropbox (Cambridge University)/PhD_Georgeos_Hardo/ML_based_segmentation_results/40x_Ph2_test_1.5/output_pickles/cell_timeseries_short_trench.p", "wb") as f:
        pickle.dump(cell_timeseries, f)
    with open("/home/georgeos/Storage/Dropbox (Cambridge University)/PhD_Georgeos_Hardo/ML_based_segmentation_results/40x_Ph2_test_1.5/output_pickles/space_timeseries_short_trench.p", "wb") as f:
        pickle.dump(space, f)