import pygame
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
from symbac.simulation.cell import Cell, generate_color

"""
Initializes Pygame and Pymunk and runs the main simulation loop.
"""
pygame.init()
screen_width, screen_height = 1200, 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Dividing Worm Colony Simulation")

space = pymunk.Space(threaded=True)
space.threads = 2


def setup_spatial_hash(space, colony):
    """Setup spatial hash with estimated parameters"""
    dim, count = estimate_spatial_hash_params(colony)
    space.use_spatial_hash(dim, count)
    print(f"Spatial hash enabled: dim={dim:.1f}, count={count}")
    return dim, count


def estimate_spatial_hash_params(colony):
    """Estimate good spatial hash parameters for current colony size"""
    if not colony:
        return 36.0, 1000

    total_segments = sum(len(cell.bodies) for cell in colony)
    segment_radius = colony[0].segment_radius if colony else 15

    # dim: slightly larger than segment diameter for optimal performance
    dim = segment_radius * 2 * 1.2

    # count: ~10x total objects, with reasonable bounds
    count = max(1000, min(100000, total_segments * 10))
    print(f"Spatial hash updated: dim={dim:.1f}, count={count}")
    return dim, count

# Add periodic updates in your main loop:
frame_count = 0
last_segment_count = 15  # Initial worm segments

space.iterations = 60
space.gravity = (0, 0)
space.damping = 0.5
# NEW: Camera and zoom variables
zoom_level = 1.0
camera_x, camera_y = 0, 0
min_zoom = 0.1
max_zoom = 5.0

# Create a virtual surface for drawing
virtual_surface = pygame.Surface((screen_width, screen_height))
draw_options = pymunk.pygame_util.DrawOptions(virtual_surface)
pymunk.pygame_util.positive_y_is_up = False

colony = []
next_group_id = 1

initial_worm = Cell(
    space,
    start_pos=(screen_width / 2, screen_height / 2),
    num_segments=15,
    segment_radius=15,
    segment_mass=2,
    group_id=next_group_id,
    base_max_length=60,  # This is now the mean length
    noise_strength=0.1,  # NEW: Small environmental noise
)
colony.append(initial_worm)
next_group_id += 1

# In your main() function, after creating initial_worm:
setup_spatial_hash(space, colony)

mouse_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
mouse_joint = None

clock = pygame.time.Clock()
running = True

simulation_speed_multiplier = 10

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
                if hasattr(draw_options, 'show_joints'):
                    draw_options.show_joints = not draw_options.show_joints

        elif event.type == pygame.MOUSEBUTTONDOWN:
            # NEW: Convert mouse position to world coordinates
            mouse_x, mouse_y = event.pos
            world_x = (mouse_x - screen_width / 2) / zoom_level + camera_x
            world_y = (mouse_y - screen_height / 2) / zoom_level + camera_y
            pos = Vec2d(world_x, world_y)

            hit = space.point_query_nearest(pos, 5 / zoom_level, pymunk.ShapeFilter())
            if hit is not None and hit.shape.body.body_type == pymunk.Body.DYNAMIC:
                shape = hit.shape
                rest_point = shape.body.world_to_local(pos)
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

    # In sim_loop.py, find this part of the main while loop


    for _ in range(simulation_speed_multiplier):
        newly_born_worms_map = {}

        for worm in colony[:]:
            worm.apply_noise(dt)
            worm.grow(dt)
            # --- MODIFY THIS LINE ---
            new_worm = worm.divide(next_group_id, dt)  # Pass dt here
            # --- END OF MODIFICATION ---
            if new_worm:
                newly_born_worms_map[new_worm] = worm
                next_group_id += 1

        # ... (rest of the loop remains the same)

        if newly_born_worms_map:
            counter = 0
            for daughter, mother in newly_born_worms_map.items():
                while True:
                    overlap_found = False
                    for daughter_shape in daughter.shapes:
                        query_result = space.shape_query(daughter_shape)

                        for info in query_result:
                            if info.shape in mother.shapes:
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

        colony.extend(newly_born_worms_map.keys())
        space.step(dt)

    # NEW: Apply camera transform to draw options
    draw_options.transform = pymunk.Transform(
        a=zoom_level, b=0, c=0, d=zoom_level,
        tx=screen_width / 2 - camera_x * zoom_level,
        ty=screen_height / 2 - camera_y * zoom_level
    )

    # NEW: Hide joints when zoomed out, show when zoomed in
    if zoom_level < 0.8:  # Hide joints when zoomed out
        draw_options.flags = pymunk.pygame_util.DrawOptions.DRAW_SHAPES
    else:  # Show joints when zoomed in enough
        draw_options.flags = (
                pymunk.pygame_util.DrawOptions.DRAW_SHAPES |
                pymunk.pygame_util.DrawOptions.DRAW_CONSTRAINTS
        )

    # Clear both surfaces
    screen.fill((20, 30, 40))
    virtual_surface.fill((20, 30, 40))

    # Draw to virtual surface with transform
    space.debug_draw(draw_options)

    # Blit virtual surface to main screen
    screen.blit(virtual_surface, (0, 0))

    # NEW: Display zoom level and controls
    font = pygame.font.Font(None, 36)
    zoom_text = font.render(f"Zoom: {zoom_level:.2f}x", True, (255, 255, 255))
    help_text = font.render("Mouse wheel: Zoom, R: Reset, J: Toggle joints", True, (255, 255, 255))
    screen.blit(zoom_text, (10, 10))
    screen.blit(help_text, (10, 50))

    pygame.display.flip()
    clock.tick(60)

    frame_count += 1

    # Update spatial hash every 60 frames (1 second) if colony grew significantly
    if frame_count % 60 == 0:
        current_segment_count = sum(len(worm.bodies) for worm in colony)
        if current_segment_count > last_segment_count * 1.5:  # 50% growth
            dim, count = estimate_spatial_hash_params(colony)
            space.use_spatial_hash(dim, count)
            last_segment_count = current_segment_count

pygame.quit()