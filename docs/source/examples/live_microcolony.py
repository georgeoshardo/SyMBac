"""Live free-growing microcolony (no walls) using the SyMBac segment-chain physics.

A single seed cell grows into a colony on an open plane, as on an agar pad. Cell
physics and hooks mirror what ``Simulation.run_simulation`` sets up for the mother
machine, minus the trench geometry: compression-dependent growth rate, width
relaxation, and a division length resampled each generation.

Requires pyqtgraph and a Qt binding. Run from the repository root:

    pixi run python docs/source/examples/live_microcolony.py [MAX_CELLS]

MAX_CELLS defaults to 600. The simulation pauses once the colony reaches it; close
the window or press Esc to exit.
"""
import sys
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from SyMBac.physics.config import CellConfig, PhysicsConfig
from SyMBac.physics.simulator import Simulator

MAX_CELLS = int(sys.argv[1]) if len(sys.argv) > 1 else 600
STEPS_PER_FRAME = 10

np.random.seed(0)

# Same unit conversion as the example notebook: 0.065 um/px, resize_amount=3.
scale = 3 / 0.065
cell_width_um, cell_max_length_um = 1.0, 3.5
R = cell_width_um * scale / 2
G = 4
radius_scale = R / 10

cell_config = CellConfig(
    GRANULARITY=G,
    SEGMENT_RADIUS=R,
    SEGMENT_MASS=1.0,
    GROWTH_RATE=0.5 * R,
    MIN_LENGTH_AFTER_DIVISION=max(3, G),
    BASE_MAX_LENGTH=cell_max_length_um * scale,
    MAX_LENGTH_STD=0.3 * scale,
    WIDTH_STD=0.05 * scale,
    SEED_CELL_SEGMENTS=max(3, int(cell_max_length_um * 0.5 * scale / (R / G))),
    PIVOT_JOINT_STIFFNESS=5_000 * radius_scale,
    NOISE_STRENGTH=0.05,
    START_POS=(0.0, 0.0),
    START_ANGLE=0.3,
    SEPTUM_DURATION=1.5,
    ROTARY_LIMIT_JOINT=True,
    MAX_BEND_ANGLE=0.005,
    STIFFNESS=300_000 * radius_scale,
    SIMPLE_LENGTH=False,
)
physics_config = PhysicsConfig(ITERATIONS=15 * 8, DAMPING=0.5, GRAVITY=(0, 0))


def cell_growth_rate_updater(cell):
    cell.update_width_transition(physics_config.DT)
    compression_ratio = cell.physics_representation.get_compression_ratio()
    cell.adjusted_growth_rate = cell.config.GROWTH_RATE * compression_ratio ** 4


def post_growth_width_sync(cell):
    cell.apply_current_width_to_segments()


def resample_max_lengths_after_division(mother, daughter):
    mother.max_length = mother.sample_max_length()
    daughter.max_length = daughter.sample_max_length()


sim = Simulator(
    physics_config=physics_config,
    initial_cell_config=cell_config,
    pre_cell_grow_hooks=[cell_growth_rate_updater],
    post_cell_grow_hooks=[post_growth_width_sync],
    post_division_hooks=[resample_max_lengths_after_division],
)

# --- Viewer ---
app = pg.mkQApp("SyMBac microcolony")
win = pg.GraphicsLayoutWidget(show=True, title="SyMBac microcolony")
win.resize(900, 900)
win.setBackground("k")
win.keyPressEvent = lambda event: win.close() if event.key() == QtCore.Qt.Key.Key_Escape else None
plot = win.addPlot()
plot.hideAxis("left")
plot.hideAxis("bottom")
plot.setAspectLocked(True)
scatter = pg.ScatterPlotItem(pxMode=False, pen=None)
plot.addItem(scatter)

_brush_cache = {}


def brush_for(group_id):
    brush = _brush_cache.get(group_id)
    if brush is None:
        rng = np.random.default_rng(group_id)
        hue = rng.random()
        brush = pg.mkBrush(pg.hsvColor(hue, sat=0.55, val=0.95))
        _brush_cache[group_id] = brush
    return brush


def redraw():
    xs, ys, sizes, brushes = [], [], [], []
    for cell in sim.cells:
        brush = brush_for(cell.group_id)
        for segment in cell.physics_representation.segments:
            x, y = segment.body.position
            xs.append(x)
            ys.append(y)
            sizes.append(2 * segment.radius)
            brushes.append(brush)
    scatter.setData(x=xs, y=ys, size=sizes, brush=brushes)
    if xs:
        pad = 4 * R
        plot.setRange(xRange=(min(xs) - pad, max(xs) + pad), yRange=(min(ys) - pad, max(ys) + pad), padding=0)


t0 = time.perf_counter()


def tick():
    if sim.num_cells >= MAX_CELLS:
        timer.stop()
        win.setWindowTitle(f"SyMBac microcolony: done, {sim.num_cells} cells (close window to exit)")
        print(f"Reached {sim.num_cells} cells in {time.perf_counter() - t0:.1f} s ({sim.frame_count} steps)")
        return
    for _ in range(STEPS_PER_FRAME):
        sim.step()
    redraw()
    win.setWindowTitle(
        f"SyMBac microcolony: {sim.num_cells} cells, step {sim.frame_count}, {time.perf_counter() - t0:.0f} s"
    )


redraw()
timer = QtCore.QTimer()
timer.timeout.connect(tick)
timer.start(0)
app.exec()
