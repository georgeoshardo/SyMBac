import sys
sys.path.insert(0,'/home/georgeos/Documents/GitHub/SYMPTOMM2')
import pymunk
import matplotlib.pyplot as plt
import numpy as np
from SYMPTOMM import cell_geometry
from SYMPTOMM.cell import Cell
from SYMPTOMM.scene_functions import create_space, step_and_update
from SYMPTOMM.plotting import plot_scene
from SYMPTOMM.trench_geometry import trench_creator
import pyglet
import pymunk
from pymunk.pyglet_util import DrawOptions

window = pyglet.window.Window(180, 700, "MM Trench test", resizable=False)
options = DrawOptions()
space = create_space()

space.gravity = 0, -10
dt = 1/100

trench_creator(45,600,(0,0), space)
trench_creator(45,600,(45*2,0), space)
#trench_creator(45,600,(45*4,0), space)
#trench_creator(45,600,(45*4,0), space)
#trench_creator(45,600,(45*6,0), space)

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
    position = (100,100), 
    angle = np.pi/3, 
    space = space,
    dt = 1/60,
    growth_rate_constant = 1,
    max_length_mean = 80,
    max_length_var = 0,
)
cell3 = Cell(
    length = 40, 
    width = 20, 
    resolution = 20, 
    position = (200,100), 
    angle = np.pi/3, 
    space = space,
    dt = 1/60,
    growth_rate_constant = 1,
    max_length_mean = 80,
    max_length_var = 0,
)
cell4 = Cell(
    length = 40, 
    width = 20, 
    resolution = 20, 
    position = (300,100), 
    angle = np.pi/3, 
    space = space,
    dt = 1/60,
    growth_rate_constant = 1,
    max_length_mean = 80,
    max_length_var = 0,
)
cells = [cell1]






@window.event
def on_draw():
    window.clear()
    space.debug_draw(options)

      




if __name__ == "__main__":
    pyglet.clock.schedule_interval(step_and_update, dt, cells, space, 30)
    pyglet.app.run()