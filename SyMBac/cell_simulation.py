#%%
import pyglet
import pymunk as pm
from pymunk.pyglet_util import DrawOptions
import time
import numpy as np
from dataclasses import dataclass
from typing import Any
import math
from joblib import Parallel, delayed
from numba import njit
from numba.experimental import jitclass
from copy import deepcopy
from cell import Cell



cell_1 = Cell(length = 15, width = 7.5, y_pos = 768/4, x_pos = 1024/4, angle=2*np.pi*np.random.rand())

space = pm.Space()
options = DrawOptions()
options.collision_point_color = (1,1,0,0)
space.debug_draw(options)
space.gravity = 0, 0
cells =[cell_1]
for cell in cells:
    space.add(cell.pm_object[0], cell.pm_object[1])

    
    
def space_updater(dt):
    for shape in space.shapes:
        space.remove(shape)
    for body in space.bodies:
        space.remove(body)


    for cell in cells:
        cell.growth(np.random.uniform(low = 0, high = 0.1))
        cell.update_pm_object()
        if cell.is_dividing() == 1:
            cell.divide()
            cell.update_pm_object()
            daughter = Cell(length = cell.daughter_length(), width = cell.daughter_width(), x_pos = cell.daughter_x_pos(), y_pos = cell.daughter_y_pos(), angle=cell.daughter_angle())
            cells.append(daughter)
        cell_adder(cell)
        if len(cells) <= 20:
            for x in range(150):
                space.step(dt)
        elif len(cells) <= 100:
            for x in range(50):
                space.step(dt)
        elif len(cells) > 100:
            space.step(dt)

def get_draw_params():
    for cell in cells:
        print(cell.angle, cell.length, cell.width)


def cell_updater(cell):
    cell.growth(np.random.uniform(low = 0, high = 0.2))
    cell.update_pm_object()
    if cell.is_dividing() == 1:
        cell.divide()
        cell.update_pm_object()
        daughter = Cell(length = cell.daughter_properties.length(), width = cell.daughter_properties.width(), x_pos = cell.daughter_properties.x_pos(), y_pos = cell.daughter_properties.y_pos(), angle=cell.daughter_properties.angle())
        cells.append(daughter)

def cell_adder(cell):
    space.add(cell.pm_object[0], cell.pm_object[1])


def update(dt):
    space_updater(dt)
    get_draw_params()


window = pyglet.window.Window(1024,768, "test", resizable = True)
options = DrawOptions()
@window.event
def on_draw():
    window.clear()
    space.debug_draw(options)

if __name__ == "__main__":
    pyglet.clock.schedule_interval(update, 1/100)
    pyglet.app.run()