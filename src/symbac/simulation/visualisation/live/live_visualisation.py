import pygame
import pymunk
import pymunk.pygame_util
from symbac.simulation.config import SimViewerConfig
import numpy as np
class LiveVisualisation:
    def __init__(self, sim_viewer_config: SimViewerConfig):
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
        self.show_joints = True

        # Create a virtual surface for drawing
        self.virtual_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.virtual_surface)
        pymunk.pygame_util.positive_y_is_up = False

        self.draw_options.flags = pymunk.pygame_util.DrawOptions.DRAW_SHAPES

        self.clock = pygame.time.Clock()

    def handle_input(self, simulator):

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

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # NEW: Convert mouse position to world coordinates
                mouse_x, mouse_y = event.pos
                world_x = (mouse_x - self.screen_width / 2) / self.zoom_level + self.camera_x
                world_y = (mouse_y - self.screen_height / 2) / self.zoom_level + self.camera_y
                pos = (world_x, world_y)

                hit = simulator.space.point_query_nearest(pos, 5 / self.zoom_level, pymunk.ShapeFilter())
                if hit is not None and hit.shape.body.body_type == pymunk.Body.DYNAMIC:
                    shape = hit.shape
                    rest_point = shape.body.world_to_local(pos)
                    rest_point = (rest_point[0], rest_point[1])
                    self.mouse_joint = pymunk.PivotJoint(
                        self.mouse_body, shape.body, (0, 0), rest_point
                    )
                    self.mouse_joint.max_force = 100000
                    self.mouse_joint.error_bias = (1 - 0.15) ** 60
                    simulator.space.add(self.mouse_joint)
            elif event.type == pygame.MOUSEBUTTONUP:
                if self.mouse_joint is not None:
                    simulator.space.remove(self.mouse_joint)
                    self.mouse_joint = None

        # NEW: Update mouse body position in world coordinates
        mouse_x, mouse_y = pygame.mouse.get_pos()
        world_mouse_x = (mouse_x - self.screen_width / 2) / self.zoom_level + self.camera_x
        world_mouse_y = (mouse_y - self.screen_height / 2) / self.zoom_level + self.camera_y
        self.mouse_body.position = (world_mouse_x, world_mouse_y)

    def draw(self, simulator):
        self.handle_input(simulator)
        # Only draw every 10 frames to reduce load
        if simulator.frame_count % self.sim_viewer_config.SIM_STEPS_PER_DRAW == 0:
            # Apply camera transform to draw options
            self.draw_options.transform = pymunk.Transform(
                a=self.zoom_level, b=0, c=0, d=self.zoom_level,
                tx=self.screen_width / 2 - self.camera_x * self.zoom_level,
                ty=self.screen_height / 2 - self.camera_y * self.zoom_level
            )

            if self.show_joints and self.zoom_level >= 0.8:
                self.draw_options.flags |= pymunk.pygame_util.DrawOptions.DRAW_CONSTRAINTS



            # Clear both surfaces
            self.screen.fill(self.sim_viewer_config.BACKGROUND_COLOR)
            self.virtual_surface.fill(self.sim_viewer_config.BACKGROUND_COLOR)

            # Draw to virtual surface with transform
            simulator.space.debug_draw(self.draw_options)

            # Blit virtual surface to main screen
            self.screen.blit(self.virtual_surface, (0, 0))

            # NEW: Display zoom level, and controls
            font = pygame.font.Font(None, self.sim_viewer_config.FONT_SIZE)
            zoom_text = font.render(f"Zoom: {self.zoom_level:.2f}x", True, (255, 255, 255))
            help_text = font.render("Mouse wheel: Zoom, R: Reset, J: Toggle joints", True, (255, 255, 255))
            self.screen.blit(zoom_text, (10, 10))
            self.screen.blit(help_text, (10, 90))

            pygame.display.flip()
            self.clock.tick(np.inf)