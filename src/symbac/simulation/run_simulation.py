# Example code to run a simulation with various hooks and configurations etc

from symbac.simulation.simulator import Simulator
from config import CellConfig, PhysicsConfig

physics_config = PhysicsConfig(
    THREADS=2,
    THREADED=True,
    ITERATIONS=100
)

initial_cell_config = CellConfig(
    GRANULARITY=10, # 16 is good for precise division with no gaps, 8 is a good compromise between performance and precision, 3 is for speed
    SEGMENT_RADIUS=15,
    SEGMENT_MASS=1.0,
    GROWTH_RATE=10, # Turning up the growth rate is a good way to speed up the simulation while keeping ITERATIONS high,
    BASE_MAX_LENGTH=180, # This should be stable now!
    MAX_LENGTH_VARIATION=0.24,
    MIN_LENGTH_AFTER_DIVISION=4,
    NOISE_STRENGTH=0.05,
    SEED_CELL_SEGMENTS=30,
    ROTARY_LIMIT_JOINT=True,
    MAX_BEND_ANGLE=0.005,
    STIFFNESS=300_0000 , # Common values: (bend angle = 0.005, stiffness = 300_000), you can use np.inf for max stiffness but ideally use np.iinfo(np.int64).max for integer type
    #DAMPED_ROTARY_SPRING=True,  # Enable damped rotary springs, makes cells quite rigid
    #ROTARY_SPRING_STIFFNESS=2000_000, # A good starting point
    #ROTARY_SPRING_DAMPING=200_000, # A good starting point
    PIVOT_JOINT_STIFFNESS=5000 # This can be lowered from the default np.inf, and the cell will be able to compress
)

simulator = Simulator(physics_config, initial_cell_config)

STEPS = 3000

# Create an object to log the simulation context each frame for plotting later
from tqdm.auto import tqdm
import time
class SimulationLogger:
    def __init__(self):
        # Initialize lists to store simple stats
        self.num_cells = []
        self.t = []

        # Create a tqdm progress bar instance, which we will manually update
        self.pbar = tqdm(total=STEPS, unit="step", desc="Simulation Progress", smoothing =0.0)
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
        current_time = time.time()
        # Calculate the time elapsed for this single step
        step_time_ms = (current_time - self.last_time) * 1000
        self.last_time = current_time
        self.pbar.set_postfix(cells=self.num_cells[-1], time_per_step=f"{step_time_ms:.2f}ms")
        # Advance the progress bar by one step
        self.pbar.update(1)

    def log_cell_positions(self, simulator: 'Simulator') -> None:
        # This function can be used to log cell positions if needed
        current_frame_data = [
            {
                'position': (seg.body.position.x, seg.body.position.y),
                'radius': seg.shape.radius,
                'id': cell.group_id
            }
            for cell in simulator.cells for seg in cell.PhysicsRepresentation.segments
        ]
        self.frames_to_draw_mpl.append((simulator.frame_count, current_frame_data))

my_logger = SimulationLogger()

simulator.add_post_step_hook(my_logger.log_frame)
simulator.add_post_step_hook(my_logger.get_step_comp_time)
simulator.add_post_step_hook(my_logger.log_cell_positions)

for step in range(STEPS):
    simulator.step()

import matplotlib.pyplot as plt
plt.plot(my_logger.t, my_logger.num_cells)
plt.show()
