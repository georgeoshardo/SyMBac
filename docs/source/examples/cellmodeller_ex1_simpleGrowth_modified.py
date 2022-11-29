import random

import numpy as np
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator

cell_cols = {0: [1, 0, 0], 1: [1, 1, 1], 2: [0, 0, 1.0], 3: [0, 0.5, 1]} # ... can add more cell colours if more cell types are added

def setup(sim):
    # Set biophysics, signalling, and regulation models
    biophys = CLBacterium(sim, jitter_z=False, max_cells=2000, max_planes=3, gamma=100.)

    # use this file for reg too
    regul = ModuleRegulator(sim, sim.moduleName)
    # Only biophys and regulation
    sim.init(biophys, regul, None, None)

    # Specify the initial cell and its location in the simulation
    sim.addCell(cellType=0, pos=(np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0),
                dir=(np.random.rand() * 3, np.random.rand() * 3, 0))

    # Add some objects to draw the models
    # if sim.is_gui:
    from CellModeller.GUI import Renderers
    therenderer = Renderers.GLBacteriumRenderer(sim)
    sim.addRenderer(therenderer)


def init(cell):
    # Specify mean and distribution of initial cell size
    cell.targetVol = 3.5 + random.uniform(-0.25, 0.5)
    # Specify growth rate of cells
    cell.growthRate = 1
    cell.color = cell_cols[cell.cellType]
    cell.killFlag = False
    cell_geom = (2000,)
    cell.force = np.zeros(cell_geom)
    #cell.radius = 4


def update(cells):
    # Iterate through each cell and flag cells that reach target size for division
    for (id, cell) in cells.items():
        if cell.length > cell.targetVol:
            cell.divideFlag = True
        else:
            p = np.array(cell.pos)
            r = np.sqrt(np.sum(p * p))
            # Remove cells from the simulation if they are more than 15 units from the centre.
            #if (abs(cell.pos[1]) > 15):
            #    cell.killFlag = True
            if r > 25:
                cell.killFlag = True
        gr = cell.strainRate / 0.05
        cgr = gr - 0.5

def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    d1.targetVol = 2.5 + random.uniform(0.0, 4.5)
    d2.targetVol = 2.5 + random.uniform(0.0, 4.5)