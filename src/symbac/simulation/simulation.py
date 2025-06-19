import pymunk
from symbac.simulation.cell import Cell
from symbac.simulation.config import CellConfig, PhysicsConfig

space = pymunk.Space(threaded=PhysicsConfig.THREADED)
space.threads = PhysicsConfig.THREADS
space.iterations = PhysicsConfig.ITERATIONS
space.gravity = PhysicsConfig.GRAVITY
space.damping = PhysicsConfig.DAMPING



def setup_spatial_hash(space: pymunk.Space, colony: list) -> tuple[float, int]:
    """Setup spatial hash with estimated parameters"""
    dim, count = estimate_spatial_hash_params(colony)
    space.use_spatial_hash(dim, count)
    #print(f"Spatial hash enabled: dim={dim:.1f}, count={count}")
    return dim, count


def estimate_spatial_hash_params(colony: list[Cell]) -> tuple[float, int]:
    """Estimate good spatial hash parameters for current colony size"""
    if not colony:
        return 36.0, 1000

    total_segments = sum(len(cell.segments) for cell in colony)
    segment_radius = colony[0].config.SEGMENT_RADIUS if colony else 15

    # dim: slightly larger than segment diameter for optimal performance
    dim = segment_radius * 2 * 1.2

    # count: ~10x total objects, with reasonable bounds
    count = max(1000, min(100000, total_segments * 10))
    return dim, count


# Add periodic updates in your main loop:
frame_count = 0
image_count = 0
last_segment_count = 15  # Initial cell segments
colony: list[Cell] = []
next_group_id = 1

initial_cell_config = CellConfig(
    GRANULARITY=3,
    SEGMENT_RADIUS=15,
    SEGMENT_MASS=1.0,
    GROWTH_RATE=2,
    BASE_MAX_LENGTH=55, # This should be stable now!
    MAX_LENGTH_VARIATION=0.5,
    MIN_LENGTH_AFTER_DIVISION=10,
    NOISE_STRENGTH=0.05,
    SEED_CELL_SEGMENTS=30,
    ROTARY_LIMIT_JOINT=True,
    MAX_BEND_ANGLE=0.01,
    STIFFNESS=300_000 , # Common values: (bend angle = 0.005, stiffness = 300_000), you can use np.inf for max stiffness but ideally use np.iinfo(np.int64).max for integer type
    #DAMPED_ROTARY_SPRING=True,  # Enable damped rotary springs, makes cells quite rigid
    #ROTARY_SPRING_STIFFNESS=2000_000, # A good starting point
    #ROTARY_SPRING_DAMPING=200_000, # A good starting point
    #PIVOT_JOINT_STIFFNESS=np.inf # This can be lowered from the default np.inf, and the cell will be able to compress
)

initial_cell = Cell(
    space,
    config=initial_cell_config,
    start_pos=(0, 0),
    group_id=next_group_id,
)
colony.append(initial_cell)
next_group_id += 1

setup_spatial_hash(space, colony)

import os
output_directory = "frames"
os.makedirs(output_directory, exist_ok=True)
print(f"Saving debug frames to ./{output_directory}/")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os


def draw_colony_matplotlib(colony: list[Cell], frame_number: int, output_dir: str = "frames"):
    """
    Draws the current state of the colony using Matplotlib and saves it to a file.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Dynamically set plot limits based on colony bounds, with a buffer
    if not any(cell.segments for cell in colony):
        # Default view if colony is empty
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
    else:
        all_positions = np.array([
            seg.body.position for cell in colony for seg in cell.segments
        ])
        min_coords = all_positions.min(axis=0)
        max_coords = all_positions.max(axis=0)
        center = (min_coords + max_coords) / 2
        # Determine the required viewing range, add a 20% buffer and a fixed minimum size
        view_range = (max_coords - min_coords).max() * 1.2 + 200
        ax.set_xlim(center[0] - view_range / 2, center[0] + view_range / 2)
        ax.set_ylim(center[1] - view_range / 2, center[1] + view_range / 2)

    # Draw each segment
    for cell in colony:
        for segment in cell.segments:
            x, y = segment.body.position
            r = segment.shape.radius

            # Pymunk colors are (R, G, B, A) in the 0-255 range.
            # Matplotlib's Circle patch expects RGB(A) in the 0-1 range.
            rgba_color = np.array(segment.shape.color) / 255.0

            circle = patches.Circle((x, y), radius=r, color=rgba_color)
            ax.add_patch(circle)

    # Configure plot appearance
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Colony State at Frame {frame_number}")
    ax.set_facecolor('black')

    # Save the figure to the specified directory
    output_path = os.path.join(output_dir, f"frame_{frame_number:05d}.jpg")
    plt.savefig(output_path)
    plt.close(fig)  # This is crucial to prevent consuming all memory

while True:

    dt = 1.0 / 60.0


    newly_born_cells_map = {}

    for cell in colony[:]:
        cell.apply_noise(dt)
        cell.grow(dt)
        new_cell = cell.divide(next_group_id, dt)  # Pass dt here
        if new_cell:
            newly_born_cells_map[new_cell] = cell
            next_group_id += 1
            print(newly_born_cells_map)

    if newly_born_cells_map:
        counter = 0
        for daughter, mother in newly_born_cells_map.items():
            # Create a set of the mother's shapes for efficient lookup
            mother_shapes = {s.shape for s in mother.segments}
            while True:
                overlap_found = False
                for daughter_segment in daughter.segments:
                    query_result = space.shape_query(daughter_segment.shape)

                    for info in query_result:
                        if info.shape in mother_shapes:
                            mother.remove_tail_segment()
                            overlap_found = True
                            break
                    if overlap_found:
                        break

                if not overlap_found:
                    break
                counter += 1
                if counter > 100:
                    break

    colony.extend(newly_born_cells_map.keys())
    space.step(dt)
    frame_count += 1

    # Update spatial hash every 60 steps
    if frame_count % 20 == 0:
        current_segment_count = sum(len(cell.segments) for cell in colony)
        if current_segment_count > last_segment_count * 1.5:  # 50% growth
            dim, count = estimate_spatial_hash_params(colony)
            space.use_spatial_hash(dim, count)
            last_segment_count = current_segment_count
        draw_colony_matplotlib(colony, image_count, output_directory)
        image_count += 1
        print("Drew frame", frame_count, "with", current_segment_count, "segments.")
