import numpy as np
np.random.seed(42)
import pygame
import pymunk.pygame_util
from cell import Cell
from config import CellConfig, SimViewerConfig, PhysicsConfig
import numpy as np

"""
Initializes Pygame and Pymunk and runs the main simulation loop.
"""
pygame.init()


screen_width, screen_height = SimViewerConfig.SCREEN_WIDTH, SimViewerConfig.SCREEN_HEIGHT
simulation_speed_multiplier = SimViewerConfig.SIMULATION_SPEED_MULTIPLIER
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Cell Colony Simulation")

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

colony: list[Cell] = []
next_group_id = 1

initial_cell_config = CellConfig(
    GRANULARITY=6, # 16 is good for precise division with no gaps, 8 is a good compromise between performance and precision, 3 is for speed
    SEGMENT_RADIUS=15,
    SEGMENT_MASS=1.0,
    GROWTH_RATE=10, # Turning up the growth rate is a good way to speed up the simulation while keeping ITERATIONS high,
    BASE_MAX_LENGTH=40, # This should be stable now!
    MAX_LENGTH_VARIATION=0.5,
    MIN_LENGTH_AFTER_DIVISION=4,
    NOISE_STRENGTH=0.05,
    SEED_CELL_SEGMENTS=30,
    ROTARY_LIMIT_JOINT=True,
    MAX_BEND_ANGLE=0.005,
    STIFFNESS=300_0000 , # Common values: (bend angle = 0.005, stiffness = 300_000), you can use np.inf for max stiffness but ideally use np.iinfo(np.int64).max for integer type
    #DAMPED_ROTARY_SPRING=True,  # Enable damped rotary springs, makes cells quite rigid
    #ROTARY_SPRING_STIFFNESS=2000_000, # A good starting point
    #ROTARY_SPRING_DAMPING=200_000, # A good starting point
    PIVOT_JOINT_STIFFNESS=np.inf # This can be lowered from the default np.inf, and the cell will be able to compress
)

initial_cell = Cell(
    space,
    config=initial_cell_config,
    start_pos=(0, 0),
    group_id=next_group_id,
)
colony.append(initial_cell)
next_group_id += 1

# In your main() function, after creating initial_cell:
setup_spatial_hash(space, colony)

mouse_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
mouse_joint = None

clock = pygame.time.Clock()
running = True
show_joints = True


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

    dt = 1.0 / 60.0

    # NEW: Only run simulation if not paused
    if not is_paused:
        for _ in range(simulation_speed_multiplier):
            newly_born_cells_map = {}

            for cell in colony[:]:
                if frame_count % 60 == 0:
                    cell.apply_noise(dt)
                cell.check_total_compression(compression_threshold=0.1)
                cell.grow(dt)

                new_cell = cell.divide(next_group_id, dt)  # Pass dt here
                if new_cell:
                    newly_born_cells_map[new_cell] = cell
                    next_group_id += 1

            if newly_born_cells_map:
                colony.extend(newly_born_cells_map.keys())
                for daughter, mother in newly_born_cells_map.items():
                    mother_shapes = [s.shape for s in mother.segments]

                    # Symmetrical Overlap Removal Loop
                    while True:
                        overlap_found = False
                        # We only need to check the daughter's head against the mother's body
                        if daughter.segments:
                            daughter_head = daughter.segments[0]
                            query_result = space.shape_query(daughter_head.shape)

                            for info in query_result:
                                # If the daughter's head is overlapping with the mother
                                if info.shape in mother_shapes:
                                    mother.remove_tail_segment()
                                    daughter.remove_head_segment()
                                    # We must update the mother_shapes list since a shape was removed
                                    mother_shapes.pop()  # TODO this could be an issue leading to an infinite loop if the mother has no segments left and the minimum length is too high

                                    overlap_found = True
                                    break  # Exit the inner query loop

                        # If we went through a full check without finding an overlap, we're done.
                        if not overlap_found:
                            break

            space.step(dt)

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
    screen.fill(SimViewerConfig.BACKGROUND_COLOR)
    virtual_surface.fill(SimViewerConfig.BACKGROUND_COLOR)

    # Draw to virtual surface with transform
    space.debug_draw(draw_options)

    # Blit virtual surface to main screen
    screen.blit(virtual_surface, (0, 0))

    # NEW: Display zoom level, pause status, and controls
    font = pygame.font.Font(None, SimViewerConfig.FONT_SIZE)
    zoom_text = font.render(f"Zoom: {zoom_level:.2f}x", True, (255, 255, 255))
    # NEW: Show pause status
    pause_text = font.render(f"{'PAUSED' if is_paused else 'Running'}", True, (255, 255, 0) if is_paused else (0, 255, 0))
    help_text = font.render("Mouse wheel: Zoom, R: Reset, J: Toggle joints, P: Pause", True, (255, 255, 255))
    screen.blit(zoom_text, (10, 10))
    screen.blit(pause_text, (10, 50))
    screen.blit(help_text, (10, 90))

    pygame.display.flip()
    clock.tick(60000)

    # NEW: Only increment frame counter and update spatial hash if not paused
    if not is_paused:
        frame_count += 1

        # Update spatial hash every 60 frames (1 second) if colony grew significantly
        if frame_count % 60 == 0:
            current_segment_count = sum(len(cell.segments) for cell in colony)
            if current_segment_count > last_segment_count * 1.5:  # 50% growth
                dim, count = estimate_spatial_hash_params(colony)
                space.use_spatial_hash(dim, count)
                last_segment_count = current_segment_count

pygame.quit()