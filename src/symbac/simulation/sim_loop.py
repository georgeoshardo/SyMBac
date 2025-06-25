# Use sudo py-spy top --pid $(ps aux | grep python | grep -v grep | awk '{print $2, $3}' | sort -rn -k2 | head -1 | awk '{print $1}') for profiling
import colorsys
import numpy as np
from pymunk import Vec2d

from symbac.misc import generate_color
from symbac.simulation.simulator import Simulator
from symbac.simulation.visualisation.live.live_visualisation import LiveVisualisation
from symbac.trench_geometry import trench_creator

np.random.seed(42)
from simcell import SimCell
from config import CellConfig, SimViewerConfig, PhysicsConfig
import numpy as np
from tqdm import tqdm
"""
Initializes Pygame and Pymunk and runs the main simulation loop.
"""



mouse_joint = None


sim_viewer_config = SimViewerConfig(SIM_STEPS_PER_DRAW=10)
live_visualisation = LiveVisualisation(sim_viewer_config)

physics_config = PhysicsConfig(
    THREADED=True,
    THREADS=2,
    ITERATIONS=60,
)

initial_cell_config = CellConfig(
    GRANULARITY=4, # 16 is good for precise division with no gaps, 8 is a good compromise between performance and precision, 3 is for speed
    SEGMENT_RADIUS=15,
    SEGMENT_MASS=1.0,
    GROWTH_RATE=2, # Turning up the growth rate is a good way to speed up the simulation while keeping ITERATIONS high,
    BASE_MAX_LENGTH=45, # This should be stable now!
    MAX_LENGTH_VARIATION=0.24,
    MIN_LENGTH_AFTER_DIVISION=4,
    NOISE_STRENGTH=0.05,
    SEED_CELL_SEGMENTS=30,
    ROTARY_LIMIT_JOINT=True,
    MAX_BEND_ANGLE=0.005,
    START_POS=Vec2d(0, 0),
    START_ANGLE=np.pi/2,
    STIFFNESS=300_000 , # Common values: (bend angle = 0.005, stiffness = 300_000), you can use np.inf for max stiffness but ideally use np.iinfo(np.int64).max for integer type
    #DAMPED_ROTARY_SPRING=True,  # Enable damped rotary springs, makes cells quite rigid
    #ROTARY_SPRING_STIFFNESS=2000_000, # A good starting point
    #ROTARY_SPRING_DAMPING=200_000, # A good starting point
    PIVOT_JOINT_STIFFNESS=np.inf # This can be lowered from the default np.inf, and the cell will be able to compress
)

simulator = Simulator(physics_config, initial_cell_config)

import pymunk
def segment_creator(local_xy1, local_xy2, global_xy, thickness):
    segment_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    segment_shape = pymunk.Segment(segment_body, local_xy1, local_xy2, thickness)
    segment_body.position = global_xy
    segment_shape.friction = 0
    return segment_body, segment_shape

trench_creator(50, 1000, (-25, -100), simulator.space)

class CellColor:
    def __init__(self):
        self.cell_colors = {
            1: generate_color(1) # Map a cell ID to a color
        }

    def get_daughter_colour(self, mother_cell: SimCell, daughter_cell: SimCell) -> tuple[int, int, int]:
        """
        Returns a color for the daughter cell based on the mother's color.
        This can be customized to implement different inheritance strategies.
        """
        # 1. Get the mother's color and normalize it to the 0-1 range for colorsys
        r, g, b = self.cell_colors[mother_cell.group_id]
        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0

        # 2. Convert RGB to HSV
        h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
        # 3. Mutate the Hue to change the color while preserving lineage
        #    A small hue shift changes the color along the color wheel (e.g., red -> orange)
        hue_shift = np.random.uniform(-1, 1) / (np.sqrt(daughter_cell.group_id) * 2)  # Shift hue with biased rw
        new_h = (h + hue_shift) % 1.0  # Use modulo to wrap around the color wheel
        #    This prevents colors from becoming grayish or dark.
        #    We'll clamp them to a minimum vibrancy level.
        new_s = s
        new_v = v

        # 5. Convert the new HSV color back to RGB
        new_r, new_g, new_b = colorsys.hsv_to_rgb(new_h, new_s, new_v)

        # 6. Scale back to 0-255 and create the final tuple
        daughter_color = (int(new_r * 255), int(new_g * 255), int(new_b * 255))
        return daughter_color

    def update_colour(self, cell) -> None:
        a = 255
        r, g, b = self.cell_colors[cell.group_id]

        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0

        # 2. Convert RGB to HSV
        h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
        new_s = max(s / np.sqrt(cell.num_divisions+1), 0.3)  # Ensure saturation is not too low
        new_v = max(v / np.sqrt(cell.num_divisions+1), 0.3) # Ensure brightness is not too low

        # 5. Convert the new HSV color back to RGB
        r, g, b = colorsys.hsv_to_rgb(h, new_s, new_v)
        r, g, b = (int(r * 255), int(g * 255), int(b * 255))

        body_color = (r,g,b,a)
        head_color = (min(255, int(r * 1.3)), min(255, int(g * 1.3)), min(255, int(b * 1.3)), a)
        tail_color = (int(r * 0.7), int(g * 0.7), int(b * 0.7), a)
        for segment in cell.PhysicsRepresentation.segments: # You have to set a color attribute for pygame
            segment.shape.color = body_color
        cell.PhysicsRepresentation.segments[0].shape.color = head_color
        cell.PhysicsRepresentation.segments[-1].shape.color = tail_color

    def update_daughter_colour(self, mother_cell: SimCell, daughter_cell: SimCell):
        """
        Update the daughter cell's color based on the mother's color.
        This is called after a division occurs.
        """
        daughter_colour = self.get_daughter_colour(mother_cell, daughter_cell)
        self.cell_colors[daughter_cell.group_id] = daughter_colour
        self.update_colour(daughter_cell)

cell_colouriser = CellColor()

simulator.add_post_cell_iter_hook(cell_colouriser.update_colour)
simulator.add_post_division_hook(cell_colouriser.update_daughter_colour)

# Create an object to log the simulation context each frame for plotting later
from tqdm.auto import tqdm
import time
class SimulationLogger:
    def __init__(self):
        # Initialize lists to store simple stats
        self.num_cells = []
        self.t = []

        # Create a tqdm progress bar instance, which we will manually update
        self.pbar = tqdm(total=1000, unit="step", desc="Simulation Progress", smoothing =0.0)
        self.last_time = time.time()

        self.frames_to_draw_mpl = []  # Will log the positions of cells at each frame

    # A function to log the number of cells and time at each frame
    def log_frame(self, simulator: 'Simulator') -> None:
        self.num_cells.append(simulator.num_cells)
        if not self.t:
            self.t.append(0)
        else:
            self.t.append(self.t[-1] + simulator.dt)

    # A function to log the time taken for each step
    def get_step_comp_time(self, simulator: 'Simulator') -> None:
        # Don't update the bar every step, only every 20 steps
        update_interval = 20
        if simulator.frame_count % update_interval != 0:
            return
        current_time = time.time()
        # Calculate the time elapsed for this single step
        step_time_ms = (current_time - self.last_time) * 1000 / update_interval  # Convert to milliseconds and average over the last 20 steps
        self.last_time = current_time
        self.pbar.set_postfix(cells=self.num_cells[-1], time_per_step=f"{step_time_ms:.2f}ms")
        # Advance the progress bar by one step
        self.pbar.update(update_interval)

    def log_cell_positions(self, simulator: 'Simulator') -> None:
        # Log the positions of all cell segments every 100 frames
        if simulator.frame_count % 100 == 0:
            current_frame_data = [
                {
                    'position': (seg.body.position.x, seg.body.position.y),
                    'radius': seg.shape.radius,
                    'id': cell.group_id,
                    'color': generate_color(cell.group_id),
                }
                for cell in simulator.cells for seg in cell.PhysicsRepresentation.segments
            ]
            self.frames_to_draw_mpl.append((simulator.frame_count, current_frame_data))

my_logger = SimulationLogger()

simulator.add_post_step_hook(my_logger.log_frame)
simulator.add_post_step_hook(my_logger.get_step_comp_time)
simulator.add_post_step_hook(my_logger.log_cell_positions)

simulator.add_post_step_hook(live_visualisation.draw)



frames_to_render = [] # List to store data for rendering
image_count = 0
while live_visualisation.running:
        simulator.step()

