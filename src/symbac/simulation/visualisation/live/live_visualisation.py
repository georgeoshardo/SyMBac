import pygame
import pymunk
import pymunk.pygame_util
from symbac.simulation.config import SimViewerConfig
import numpy as np
import math

class LiveVisualisation:
    def __init__(self, sim_viewer_config: SimViewerConfig):
        pygame.init()
        self.sim_viewer_config = sim_viewer_config
        self.screen_width = sim_viewer_config.SCREEN_WIDTH
        self.screen_height = sim_viewer_config.SCREEN_HEIGHT
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.zoom_level = 1.0
        self.camera_x, self.camera_y = 0, 0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.mouse_joint = None  # Initialize mouse joint
        self.mouse_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.running = True
        self.show_joints = False

        # Create a virtual surface for drawing
        self.virtual_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.virtual_surface)
        pymunk.pygame_util.positive_y_is_up = False

        self.draw_options.flags = pymunk.pygame_util.DrawOptions.DRAW_SHAPES

        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Cell Colony Simulation")

        self.grid_color_minor = (40, 50, 60)
        self.grid_color_major = (60, 70, 80)

        self.origin_color = (180, 80, 80)  # A distinct red color
        self.scale_bar_color = (255, 255, 255)
        self.ui_font = pygame.font.Font(None, 24)

        self.is_panning = False
        self.pan_start_pos = (0, 0)

    def handle_input(self, simulator):
        keys = pygame.key.get_pressed()
        space_pressed = keys[pygame.K_SPACE]
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                self.running = False
                pygame.quit()
                return
            # NEW: Handle mouse wheel for zooming
            elif event.type == pygame.MOUSEWHEEL:
                old_zoom = self.zoom_level
                if event.y > 0:  # Scroll up - zoom in
                    self.zoom_level = min(self.zoom_level * 1.1, self.max_zoom)
                elif event.y < 0:  # Scroll down - zoom out
                    self.zoom_level = max(self.zoom_level / 1.1, self.min_zoom)

                # Adjust camera to zoom towards mouse position
                mouse_x, mouse_y = pygame.mouse.get_pos()
                world_x = (mouse_x - self.screen_width / 2) / old_zoom + self.camera_x
                world_y = (mouse_y - self.screen_height / 2) / old_zoom + self.camera_y

                self.camera_x = world_x - (mouse_x - self.screen_width / 2) / self.zoom_level
                self.camera_y = world_y - (mouse_y - self.screen_height / 2) / self.zoom_level

            # NEW: Handle keyboard zoom controls as backup
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.zoom_level = min(self.zoom_level * 1.2, self.max_zoom)
                elif event.key == pygame.K_MINUS:
                    self.zoom_level = max(self.zoom_level / 1.2, self.min_zoom)
                elif event.key == pygame.K_r:  # Reset zoom and camera
                    self.zoom_level = 1.0
                    self.camera_x, self.camera_y = 0, 0
                elif event.key == pygame.K_j:  # NEW: Toggle joint visibility
                    self.show_joints = not self.show_joints


            # --- MODIFIED: MOUSEBUTTONDOWN for Space + Left-Click Pan ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left-click
                    if space_pressed:
                        # Start panning
                        self.is_panning = True
                        self.pan_start_pos = event.pos
                    else:
                        # Grab an object (existing logic)
                        mouse_x, mouse_y = event.pos
                        world_x = (mouse_x - self.screen_width / 2) / self.zoom_level + self.camera_x
                        world_y = (mouse_y - self.screen_height / 2) / self.zoom_level + self.camera_y
                        pos = (world_x, world_y)
                        hit = simulator.space.point_query_nearest(pos, 5 / self.zoom_level, pymunk.ShapeFilter())
                        if hit is not None and hit.shape.body.body_type == pymunk.Body.DYNAMIC:
                            shape = hit.shape
                            rest_point = shape.body.world_to_local(pos)
                            self.mouse_joint = pymunk.PivotJoint(
                                self.mouse_body, shape.body, (0, 0), rest_point
                            )
                            self.mouse_joint.max_force = 100000
                            self.mouse_joint.error_bias = (1 - 0.15) ** 60
                            simulator.space.add(self.mouse_joint)

            # --- MODIFIED: MOUSEMOTION for dragging ---
            elif event.type == pygame.MOUSEMOTION:
                # Check if left mouse button is held down AND space is pressed
                if event.buttons[0] and space_pressed:
                    current_pos = event.pos
                    delta_x = current_pos[0] - self.pan_start_pos[0]
                    delta_y = current_pos[1] - self.pan_start_pos[1]

                    self.camera_x -= delta_x / self.zoom_level
                    self.camera_y -= delta_y / self.zoom_level

                    self.pan_start_pos = current_pos  # Update start for next drag segment

            # --- MODIFIED: MOUSEBUTTONUP to handle releasing the pan/grab ---
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    # Stop panning
                    self.is_panning = False
                    # Release grabbed object
                    if self.mouse_joint is not None:
                        simulator.space.remove(self.mouse_joint)
                        self.mouse_joint = None

        # Update mouse body position (for grabbing)
        mouse_x, mouse_y = pygame.mouse.get_pos()
        world_mouse_x = (mouse_x - self.screen_width / 2) / self.zoom_level + self.camera_x
        world_mouse_y = (mouse_y - self.screen_height / 2) / self.zoom_level + self.camera_y
        self.mouse_body.position = (world_mouse_x, world_mouse_y)

    def draw(self, simulator):
        self.handle_input(simulator)

        if simulator.frame_count % self.sim_viewer_config.SIM_STEPS_PER_DRAW == 0:
            # Apply camera transform to draw options
            self.draw_options.transform = pymunk.Transform(
                a=self.zoom_level, b=0, c=0, d=self.zoom_level,
                tx=self.screen_width / 2 - self.camera_x * self.zoom_level,
                ty=self.screen_height / 2 - self.camera_y * self.zoom_level
            )

            # 1. Clear the main screen and draw the grid in the background
            self.screen.fill(self.sim_viewer_config.BACKGROUND_COLOR)
            self._draw_grid()

            # 2. Clear the virtual surface with a transparent color
            self.virtual_surface.fill((0, 0, 0, 0))

            # 3. Set drawing flags and draw the physics simulation to the virtual surface
            if self.show_joints and self.zoom_level >= 0.8:
                self.draw_options.flags |= pymunk.pygame_util.DrawOptions.DRAW_CONSTRAINTS
            else:
                self.draw_options.flags &= ~pymunk.pygame_util.DrawOptions.DRAW_CONSTRAINTS

            simulator.space.debug_draw(self.draw_options)

            # 4. Blit the virtual surface (with cells) onto the main screen (with the grid)
            self.screen.blit(self.virtual_surface, (0, 0))

            # 5. Display text overlays
            font = pygame.font.Font(None, self.sim_viewer_config.FONT_SIZE)
            zoom_text = font.render(f"Zoom: {self.zoom_level:.2f}x", True, (255, 255, 255))
            help_text = font.render("Mouse wheel: Zoom, R: Reset, J: Toggle joints", True, (255, 255, 255))
            self.screen.blit(zoom_text, (10, 10))
            self.screen.blit(help_text, (10, 90))

            self._draw_scale_bar()

            # 6. Update the display
            pygame.display.flip()
            self.clock.tick(np.inf)

        # (Inside the LiveVisualisation class, at the end of the _draw_grid method)

    def _draw_grid(self):


        # Define base spacing for the grid in world units
        minor_spacing = 50
        major_spacing = 200

        # Adapt grid spacing based on zoom level to avoid clutter
        while minor_spacing * self.zoom_level < 15:
            minor_spacing *= 2
            major_spacing *= 2
        # --- SOLUTION: Use integer division and prevent division by zero ---
        while minor_spacing * self.zoom_level > 50 and minor_spacing > 1:
            minor_spacing //= 2
            major_spacing //= 2

        # Ensure spacing is at least 1 to avoid errors
        minor_spacing = max(1, minor_spacing)
        major_spacing = max(1, major_spacing)

        # Calculate the visible world area
        world_left = self.camera_x - (self.screen_width / 2) / self.zoom_level
        world_right = self.camera_x + (self.screen_width / 2) / self.zoom_level
        world_top = self.camera_y - (self.screen_height / 2) / self.zoom_level
        world_bottom = self.camera_y + (self.screen_height / 2) / self.zoom_level

        # --- Draw Vertical Lines ---
        start_x = int(world_left // minor_spacing) * minor_spacing
        # --- SOLUTION: Cast all arguments of range() to int ---
        for x in range(int(start_x), int(world_right) + 1, int(minor_spacing)):
            screen_x = (x - self.camera_x) * self.zoom_level + self.screen_width / 2

            if screen_x < 0 or screen_x > self.screen_width:
                continue

            color = self.grid_color_major if x % major_spacing == 0 else self.grid_color_minor
            pygame.draw.line(self.screen, color, (screen_x, 0), (screen_x, self.screen_height), 1)

        # --- Draw Horizontal Lines ---
        start_y = int(world_top // minor_spacing) * minor_spacing
        # --- SOLUTION: Cast all arguments of range() to int ---
        for y in range(int(start_y), int(world_bottom) + 1, int(minor_spacing)):
            screen_y = (y - self.camera_y) * self.zoom_level + self.screen_height / 2

            if screen_y < 0 or screen_y > self.screen_height:
                continue

            color = self.grid_color_major if y % major_spacing == 0 else self.grid_color_minor
            pygame.draw.line(self.screen, color, (0, screen_y), (self.screen_width, screen_y), 1)

        origin_x_screen = (0 - self.camera_x) * self.zoom_level + self.screen_width / 2
        origin_y_screen = (0 - self.camera_y) * self.zoom_level + self.screen_height / 2

        # Draw only if the origin is visible on screen
        if 0 < origin_x_screen < self.screen_width and 0 < origin_y_screen < self.screen_height:
            pygame.draw.line(self.screen, self.origin_color, (origin_x_screen - 10, origin_y_screen),
                             (origin_x_screen + 10, origin_y_screen), 2)
            pygame.draw.line(self.screen, self.origin_color, (origin_x_screen, origin_y_screen - 10),
                             (origin_x_screen, origin_y_screen + 10), 2)


    # (Inside the LiveVisualisation class)

    def _draw_scale_bar(self):
        """Draws a dynamic scale bar in the bottom right corner."""

        # 1. Determine a "nice" round number for the scale bar's length in world units
        # Aim for a scale bar that is roughly 150 pixels wide
        target_world_length = 150 / self.zoom_level

        # Calculate a "nice" round number (e.g., 10, 20, 50, 100)
        magnitude = 10 ** math.floor(math.log10(target_world_length))
        residual = target_world_length / magnitude

        if residual < 2:
            scale_world_units = 1 * magnitude
        elif residual < 5:
            scale_world_units = 2 * magnitude
        else:
            scale_world_units = 5 * magnitude

        # 2. Calculate the length of the bar in pixels
        scale_pixel_length = scale_world_units * self.zoom_level

        # 3. Define position and draw the bar
        padding = 20
        bar_x = self.screen_width - scale_pixel_length - padding
        bar_y = self.screen_height - padding

        # Draw main horizontal line
        pygame.draw.line(self.screen, self.scale_bar_color, (bar_x, bar_y), (bar_x + scale_pixel_length, bar_y), 2)
        # Draw vertical ticks at each end
        pygame.draw.line(self.screen, self.scale_bar_color, (bar_x, bar_y - 5), (bar_x, bar_y + 5), 2)
        pygame.draw.line(self.screen, self.scale_bar_color, (bar_x + scale_pixel_length, bar_y - 5),
                         (bar_x + scale_pixel_length, bar_y + 5), 2)

        # 4. Draw the text label above the bar
        label_text = f"{int(scale_world_units)} µm"  # Assuming world units are micrometers
        text_surface = self.ui_font.render(label_text, True, self.scale_bar_color)
        text_x = bar_x + (scale_pixel_length - text_surface.get_width()) / 2
        text_y = bar_y - text_surface.get_height() - 5
        self.screen.blit(text_surface, (text_x, text_y))








































































