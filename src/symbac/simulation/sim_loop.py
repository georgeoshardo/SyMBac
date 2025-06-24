# Use sudo py-spy top --pid $(ps aux | grep python | grep -v grep | awk '{print $2, $3}' | sort -rn -k2 | head -1 | awk '{print $1}') for profiling
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pymunk import Vec2d

from symbac.simulation.division_manager import DivisionManager
from symbac.simulation.growth_manager import GrowthManager
from symbac.simulation.simulator import Simulator
from symbac.simulation.visualisation.colony_visualiser import ColonyVisualiser
from symbac.simulation.colony import Colony

np.random.seed(42)
import pygame
import pymunk.pygame_util
from simcell import SimCell
from config import CellConfig, SimViewerConfig, PhysicsConfig
import numpy as np
from tqdm import tqdm
"""
Initializes Pygame and Pymunk and runs the main simulation loop.
"""
pygame.init()

sim_viewer_config = SimViewerConfig()
screen_width, screen_height = sim_viewer_config.SCREEN_WIDTH, sim_viewer_config.SCREEN_HEIGHT
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Cell Colony Simulation")

physics_config = PhysicsConfig()
#space = pymunk.Space(threaded=physics_config.THREADED)
#space.threads = physics_config.THREADS
#space.iterations = physics_config.ITERATIONS
#space.gravity = physics_config.GRAVITY
#space.damping = physics_config.DAMPING
#dt = physics_config.DT


def setup_spatial_hash(space: pymunk.Space, colony: Colony) -> tuple[float, int]:
    """Setup spatial hash with estimated parameters"""
    dim, count = estimate_spatial_hash_params(colony)
    space.use_spatial_hash(dim, count)
    #print(f"Spatial hash enabled: dim={dim:.1f}, count={count}")
    return dim, count


def estimate_spatial_hash_params(colony: Colony) -> tuple[float, int]:
    """Estimate good spatial hash parameters for current colony size"""
    if not colony:
        return 36.0, 1000

    total_segments = sum(len(cell.PhysicsRepresentation.segments) for cell in colony.cells)
    segment_radius = colony.cells[0].config.SEGMENT_RADIUS if colony else 15

    # dim: slightly larger than segment diameter for optimal performance
    dim = segment_radius * 2 * 1.2

    # count: ~10x total objects, with reasonable bounds
    count = max(1000, min(100000, total_segments * 10))
    return dim, count

# Add periodic updates in your main loop:
last_segment_count = 15  # Initial cell segments


# NEW: Camera and zoom variables
zoom_level = 1.0
camera_x, camera_y = 0, 0
min_zoom = 0.1
max_zoom = 5.0

# NEW: Pause state
is_paused = False


# Create a virtual surface for drawing
virtual_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
draw_options = pymunk.pygame_util.DrawOptions(virtual_surface)
pymunk.pygame_util.positive_y_is_up = False


initial_cell_config = CellConfig(
    GRANULARITY=14, # 16 is good for precise division with no gaps, 8 is a good compromise between performance and precision, 3 is for speed
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

#next_group_id = 1
#initial_cell = SimCell(
#    space,
#    config=initial_cell_config,
#    start_pos= Vec2d(0, 0),
#    group_id=next_group_id,
#)
#colony = Colony(space, [initial_cell])
#next_group_id += 1
#growth_manager = GrowthManager()
#division_manager = DivisionManager(space, initial_cell_config)
simulator = Simulator(physics_config, initial_cell_config)

# In your main() function, after creating initial_cell:
setup_spatial_hash(simulator.space, simulator.colony)

mouse_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
mouse_joint = None

clock = pygame.time.Clock()
running = True
show_joints = True



def cell_hook(simulator: Simulator, cell: SimCell):
    if simulator.frame_count % 60 == 0:
        cell.PhysicsRepresentation.apply_noise(simulator.dt)  # Jiggle cells for randomness
    #cell.PhysicsRepresentation.check_total_compression()

def post_division_hook(simulator: Simulator, cell: SimCell, daughter_cell: SimCell):
    daughter_cell.base_color = ColonyVisualiser.get_daughter_colour(cell,simulator.next_group_id)  # and set the daughter's base colour for the visualisation
    ColonyVisualiser.update_colors(daughter_cell)  # update the colours of the cell according to rules

frames_to_render = [] # List to store data for rendering
image_count = 0
pbar = tqdm(desc="Simulating...", unit="step", smoothing=0.1, dynamic_ncols=True)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            running = False
        # NEW: Handle mouse wheel for zooming
        elif event.type == pygame.MOUSEWHEEL:
            old_zoom = zoom_level
            if event.y > 0:  # Scroll up - zoom in
                zoom_level = min(zoom_level * 1.1, max_zoom)
            elif event.y < 0:  # Scroll down - zoom out
                zoom_level = max(zoom_level / 1.1, min_zoom)

            # Adjust camera to zoom towards mouse position
            mouse_x, mouse_y = pygame.mouse.get_pos()
            world_x = (mouse_x - screen_width / 2) / old_zoom + camera_x
            world_y = (mouse_y - screen_height / 2) / old_zoom + camera_y

            camera_x = world_x - (mouse_x - screen_width / 2) / zoom_level
            camera_y = world_y - (mouse_y - screen_height / 2) / zoom_level

        # NEW: Handle keyboard zoom controls as backup
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                zoom_level = min(zoom_level * 1.2, max_zoom)
            elif event.key == pygame.K_MINUS:
                zoom_level = max(zoom_level / 1.2, min_zoom)
            elif event.key == pygame.K_r:  # Reset zoom and camera
                zoom_level = 1.0
                camera_x, camera_y = 0, 0
            elif event.key == pygame.K_j:  # NEW: Toggle joint visibility
                show_joints = not show_joints
            elif event.key == pygame.K_p:  # NEW: Toggle pause
                is_paused = not is_paused
                print(f"Simulation {'paused' if is_paused else 'resumed'}")

        elif event.type == pygame.MOUSEBUTTONDOWN:
            # NEW: Convert mouse position to world coordinates
            mouse_x, mouse_y = event.pos
            world_x = (mouse_x - screen_width / 2) / zoom_level + camera_x
            world_y = (mouse_y - screen_height / 2) / zoom_level + camera_y
            pos = (world_x, world_y)

            hit = space.point_query_nearest(pos, 5 / zoom_level, pymunk.ShapeFilter())
            if hit is not None and hit.shape.body.body_type == pymunk.Body.DYNAMIC:
                shape = hit.shape
                rest_point = shape.body.world_to_local(pos)
                rest_point = (rest_point[0], rest_point[1])
                mouse_joint = pymunk.PivotJoint(
                    mouse_body, shape.body, (0, 0), rest_point
                )
                mouse_joint.max_force = 100000
                mouse_joint.error_bias = (1 - 0.15) ** 60
                space.add(mouse_joint)
        elif event.type == pygame.MOUSEBUTTONUP:
            if mouse_joint is not None:
                space.remove(mouse_joint)
                mouse_joint = None

    # NEW: Update mouse body position in world coordinates
    mouse_x, mouse_y = pygame.mouse.get_pos()
    world_mouse_x = (mouse_x - screen_width / 2) / zoom_level + camera_x
    world_mouse_y = (mouse_y - screen_height / 2) / zoom_level + camera_y
    mouse_body.position = (world_mouse_x, world_mouse_y)


    # NEW: Only run simulation if not paused
    if not is_paused:
        for _ in range(sim_viewer_config.SIM_STEPS_PER_DRAW):
            pbar.update(1)

            simulator.step()

    # NEW: Apply camera transform to draw options
    draw_options.transform = pymunk.Transform(
        a=zoom_level, b=0, c=0, d=zoom_level,
        tx=screen_width / 2 - camera_x * zoom_level,
        ty=screen_height / 2 - camera_y * zoom_level
    )

    draw_options.flags = pymunk.pygame_util.DrawOptions.DRAW_SHAPES
    if show_joints and zoom_level >= 0.8:
        draw_options.flags |= pymunk.pygame_util.DrawOptions.DRAW_CONSTRAINTS

    # Clear both surfaces
    screen.fill(sim_viewer_config.BACKGROUND_COLOR)
    virtual_surface.fill(sim_viewer_config.BACKGROUND_COLOR)

    # Draw to virtual surface with transform
    simulator.space.debug_draw(draw_options)

    # Blit virtual surface to main screen
    screen.blit(virtual_surface, (0, 0))

    # NEW: Display zoom level, pause status, and controls
    font = pygame.font.Font(None, sim_viewer_config.FONT_SIZE)
    zoom_text = font.render(f"Zoom: {zoom_level:.2f}x", True, (255, 255, 255))
    # NEW: Show pause status
    pause_text = font.render(f"{'PAUSED' if is_paused else 'Running'}", True, (255, 255, 0) if is_paused else (0, 255, 0))
    help_text = font.render("Mouse wheel: Zoom, R: Reset, J: Toggle joints, P: Pause", True, (255, 255, 255))
    screen.blit(zoom_text, (10, 10))
    screen.blit(pause_text, (10, 50))
    screen.blit(help_text, (10, 90))

    pygame.display.flip()
    clock.tick(np.inf)

    # NEW: Only increment frame counter and update spatial hash if not paused
    if not is_paused:

        # Update spatial hash every 60 frames (1 second) if colony grew significantly
        if simulator.frame_count % 1 == 0:
            current_segment_count = sum(len(cell.PhysicsRepresentation.segments) for cell in simulator.colony.cells)
            if current_segment_count > last_segment_count * 1.5:  # 50% growth
                dim, count = estimate_spatial_hash_params(simulator.colony)
                simulator.space.use_spatial_hash(dim, count)
                last_segment_count = current_segment_count

        current_segment_count = sum(len(cell.PhysicsRepresentation.segments) for cell in simulator.colony.cells)

        # Capture the state for rendering later
        current_frame_data = [
            {
                'position': (seg.body.position.x, seg.body.position.y),
                'radius': seg.shape.radius,
                'color': seg.shape.color
            }
            for cell in simulator.colony.cells for seg in cell.PhysicsRepresentation.segments
        ]
        frames_to_render.append((image_count, current_frame_data))
        image_count += 1

        if current_segment_count > last_segment_count * 1.5:
            setup_spatial_hash(space, simulator.colony)
            last_segment_count = current_segment_count


pygame.quit()
pbar.close()

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time
from tqdm import tqdm

def render_frame(frame_data: list, frame_number: int, output_dir: str):
    """
    Draws a single frame from pre-collected data using Matplotlib and saves it.
    This function is designed to be called in parallel.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))

    if not frame_data:
        ax.set_xlim(-100, 100)
        ax.set_ylim(100, -100)
    else:
        all_positions = np.array([seg['position'] for seg in frame_data])
        min_coords = all_positions.min(axis=0)
        max_coords = all_positions.max(axis=0)
        center = (min_coords + max_coords) / 2
        view_range = (max_coords - min_coords).max() * 1.2 + 200
        ax.set_xlim(center[0] - view_range / 2, center[0] + view_range / 2)
        ax.set_ylim(center[1] + view_range / 2, center[1] - view_range / 2)

    for segment_info in frame_data:
        x, y = segment_info['position']
        r = segment_info['radius']
        rgba_fill_color = np.array(segment_info['color']) / 255.0
        circle = patches.Circle((x, y), radius=r, facecolor=rgba_fill_color)
        ax.add_patch(circle)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Colony State at Frame {frame_number}")
    ax.set_facecolor('black')
    plt.axis('off')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"frame_{frame_number:05d}.jpg")
    plt.savefig(output_path)
    plt.close(fig)

# Clean output directory
output_directory = "frames"
os.makedirs(output_directory, exist_ok=True)
print(f"Clearing old frames from ./{output_directory}/")
for filename in os.listdir(output_directory):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        os.remove(os.path.join(output_directory, filename))

# --- PARALLEL RENDERING ---
if frames_to_render:
    print(f"\nStarting parallel rendering of {len(frames_to_render)} frames using all available CPU cores...")
    start_render_time = time.perf_counter()

    # Use joblib to parallelize the rendering of saved frames
    # n_jobs=-1 uses all available CPU cores
    Parallel(n_jobs=-1)(
        delayed(render_frame)(data, num, output_directory)
        for num, data in tqdm(frames_to_render, desc="Rendering frames")
    )

    end_render_time = time.perf_counter()
    print(f"Parallel rendering completed in {end_render_time - start_render_time:.2f} seconds.")
    print(f"Output frames are saved in the '{output_directory}' directory.")
