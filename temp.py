import math
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np


class Worm:
    """
    A class to create and manage a stiff, bendy worm that grows, divides,
    and is assigned a unique group ID for collision purposes.
    """

    def __init__(
        self,
        space,
        start_pos,
        num_segments,
        segment_radius,
        segment_mass,
        group_id,
        growth_rate=5.0,
        max_length=50,
        min_length_after_division=10,
        max_length_variation=0.2,
        _from_division=False,
    ):
        """
        Initializes the worm.
        The final max_length for this instance will be randomized based on the
        max_length and max_length_variation parameters.
        """
        self.space = space
        self.start_pos = start_pos
        self.segment_radius = segment_radius
        self.segment_mass = segment_mass
        self.growth_rate = growth_rate
        self.max_bend_angle = 0.01

        self.group_id = group_id

        # --- NEW: Randomize the max_length for this specific worm instance ---
        variation = max_length * max_length_variation
        random_max_len = np.random.uniform(
            max_length - variation, max_length + variation
        )
        # Ensure the length is an integer and not too small to divide
        self.max_length = max(min_length_after_division * 2, int(random_max_len))

        self.min_length_after_division = min_length_after_division
        self.max_length_variation = (
            max_length_variation  # Store for passing to offspring
        )

        self.bodies = []
        self.shapes = []
        self.joints = []

        self.growth_accumulator = 0.0
        self.growth_threshold = self.segment_radius / 3
        self.joint_distance = self.segment_radius / 4
        self.joint_max_force = 10000

        if not _from_division:
            for i in range(num_segments):
                self._add_initial_segment(i == 0)
            self._update_colors()

    def _add_initial_segment(self, is_first):
        """
        Adds a single segment to the worm during initialization.
        """
        moment = pymunk.moment_for_circle(
            self.segment_mass, 0, self.segment_radius
        )
        body = pymunk.Body(self.segment_mass, moment)

        if is_first:
            body.position = self.start_pos
        else:
            prev_body = self.bodies[-1]
            offset = Vec2d(self.joint_distance, 0).rotated(prev_body.angle)
            body.position = prev_body.position + offset

        shape = pymunk.Circle(body, self.segment_radius)
        shape.friction = 0.0  # User change
        shape.filter = pymunk.ShapeFilter(group=self.group_id)

        self.space.add(body, shape)
        self.bodies.append(body)
        self.shapes.append(shape)

        if not is_first:
            prev_body = self.bodies[-2]

            anchor_on_prev = (self.joint_distance / 2, 0)
            anchor_on_curr = (-self.joint_distance / 2, 0)
            pivot = pymunk.PivotJoint(
                prev_body, body, anchor_on_prev, anchor_on_curr
            )
            pivot.max_force = self.joint_max_force
            self.space.add(pivot)
            self.joints.append(pivot)

            limit = pymunk.RotaryLimitJoint(
                prev_body, body, -self.max_bend_angle, self.max_bend_angle
            )
            limit.max_force = self.joint_max_force
            self.space.add(limit)
            self.joints.append(limit)

    def grow(self, dt):
        """
        Grows the worm by extending the last segment until a new one can be added.
        """
        if len(self.bodies) >= self.max_length or len(self.bodies) < 2:
            return

        # User change: randomized growth
        self.growth_accumulator += (
            self.growth_rate * dt * np.random.uniform(0, 4)
        )
        last_pivot_joint = self.joints[-2]
        original_anchor_x = -self.joint_distance / 2
        last_pivot_joint.anchor_b = (
            original_anchor_x - self.growth_accumulator,
            0,
        )

        if self.growth_accumulator >= self.growth_threshold:
            pre_tail_body = self.bodies[-2]
            old_tail_body = self.bodies[-1]

            last_pivot_joint.anchor_b = (original_anchor_x, 0)

            stable_offset = Vec2d(self.joint_distance, 0).rotated(
                pre_tail_body.angle
            )
            old_tail_body.position = pre_tail_body.position + stable_offset
            old_tail_body.angle = pre_tail_body.angle

            moment = pymunk.moment_for_circle(
                self.segment_mass, 0, self.segment_radius
            )
            new_tail_body = pymunk.Body(self.segment_mass, moment)
            new_tail_offset = Vec2d(self.joint_distance, 0).rotated(
                old_tail_body.angle
            )
            new_tail_body.position = old_tail_body.position + new_tail_offset

            new_tail_shape = pymunk.Circle(new_tail_body, self.segment_radius)
            new_tail_shape.friction = 0.0  # User change
            new_tail_shape.filter = pymunk.ShapeFilter(group=self.group_id)

            self.space.add(new_tail_body, new_tail_shape)
            self.bodies.append(new_tail_body)
            self.shapes.append(new_tail_shape)

            anchor_on_prev = (self.joint_distance / 2, 0)
            anchor_on_curr = (-self.joint_distance / 2, 0)
            new_pivot = pymunk.PivotJoint(
                old_tail_body, new_tail_body, anchor_on_prev, anchor_on_curr
            )
            new_pivot.max_force = self.joint_max_force
            self.space.add(new_pivot)
            self.joints.append(new_pivot)

            new_limit = pymunk.RotaryLimitJoint(
                old_tail_body,
                new_tail_body,
                -self.max_bend_angle,
                self.max_bend_angle,
            )
            new_limit.max_force = self.joint_max_force
            self.space.add(new_limit)
            self.joints.append(new_limit)

            self.growth_accumulator = 0.0
            self._update_colors()

    def divide(self, next_group_id):
        """
        If the worm is at max_length, it splits by transplanting its second
        half into a new Cell object, preserving orientation.
        """

        if len(self.bodies) < self.max_length:
            return None

        split_index = len(self.bodies) // 2
        if split_index < self.min_length_after_division or (
            len(self.bodies) - split_index
        ) < self.min_length_after_division:
            return None

        # Create an empty "daughter" worm.
        daughter_worm = Worm(
            self.space,
            self.bodies[split_index].position,
            0,
            self.segment_radius,
            self.segment_mass,
            next_group_id,
            self.growth_rate,
            self.max_length,  # Pass on the mean max_length
            self.min_length_after_division,
            self.max_length_variation,  # Pass on the variation
            _from_division=True,
        )

        # Partition the mother's parts.
        daughter_worm.bodies = self.bodies[split_index:]
        daughter_worm.shapes = self.shapes[split_index:]
        daughter_worm.joints = self.joints[split_index * 2 :]

        # Update the collision filter for the daughter's shapes.
        for shape in daughter_worm.shapes:
            shape.filter = pymunk.ShapeFilter(group=next_group_id)

        # Find and destroy the connecting joint.
        connecting_joint = self.joints[(split_index - 1) * 2]
        connecting_limit = self.joints[(split_index - 1) * 2 + 1]
        self.space.remove(connecting_joint, connecting_limit)

        # Truncate the mother's lists.
        self.bodies = self.bodies[:split_index]
        self.shapes = self.shapes[:split_index]
        self.joints = self.joints[: (split_index - 1) * 2]

        # Update colors for both worms.
        self._update_colors()
        daughter_worm._update_colors()

        return daughter_worm

    def remove_tail_segment(self):
        """
        Safely removes the last segment of the worm.
        """
        if len(self.bodies) <= self.min_length_after_division:
            return

        tail_body = self.bodies.pop()
        tail_shape = self.shapes.pop()

        tail_joint = self.joints.pop()
        tail_limit = self.joints.pop()

        self.space.remove(tail_body, tail_shape, tail_joint, tail_limit)
        self._update_colors()

    def _update_colors(self):
        """Sets the head and tail colors."""
        if self.shapes:
            for s in self.shapes:
                s.color = (255, 100, 100, 255)
            self.shapes[0].color = (150, 255, 150, 255)
            self.shapes[-1].color = (255, 150, 150, 255)


def main():
    """
    Initializes Pygame and Pymunk and runs the main simulation loop.
    """
    pygame.init()
    screen_width, screen_height = 1200, 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Dividing Cell Colony Simulation")

    space = pymunk.Space()
    space.iterations = 60
    space.gravity = (0, 0)
    space.damping = 0.7

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

    initial_worm = Worm(
        space,
        start_pos=(screen_width / 2, screen_height / 2),
        num_segments=15,
        segment_radius=15,
        segment_mass=2,
        group_id=next_group_id,
        max_length=70,  # This is now the mean length
    )
    colony.append(initial_worm)
    next_group_id += 1

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
                world_x = (mouse_x - screen_width/2) / old_zoom + camera_x
                world_y = (mouse_y - screen_height/2) / old_zoom + camera_y
                
                camera_x = world_x - (mouse_x - screen_width/2) / zoom_level
                camera_y = world_y - (mouse_y - screen_height/2) / zoom_level
                
            # NEW: Handle keyboard zoom controls as backup
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    zoom_level = min(zoom_level * 1.2, max_zoom)
                elif event.key == pygame.K_MINUS:
                    zoom_level = max(zoom_level / 1.2, min_zoom)
                elif event.key == pygame.K_r:  # Reset zoom and camera
                    zoom_level = 1.0
                    camera_x, camera_y = 0, 0
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # NEW: Convert mouse position to world coordinates
                mouse_x, mouse_y = event.pos
                world_x = (mouse_x - screen_width/2) / zoom_level + camera_x
                world_y = (mouse_y - screen_height/2) / zoom_level + camera_y
                pos = Vec2d(world_x, world_y)
                
                hit = space.point_query_nearest(pos, 5/zoom_level, pymunk.ShapeFilter())
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
        world_mouse_x = (mouse_x - screen_width/2) / zoom_level + camera_x
        world_mouse_y = (mouse_y - screen_height/2) / zoom_level + camera_y
        mouse_body.position = (world_mouse_x, world_mouse_y)
        
        dt = 1.0 / 60.0

        for _ in range(simulation_speed_multiplier):
            newly_born_worms_map = {}

            for worm in colony[:]:
                worm.grow(dt)
                new_worm = worm.divide(next_group_id)
                if new_worm:
                    newly_born_worms_map[new_worm] = worm
                    next_group_id += 1

            if newly_born_worms_map:
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

            colony.extend(newly_born_worms_map.keys())
            space.step(dt)

        # NEW: Apply camera transform to draw options
        draw_options.transform = pymunk.Transform(
            a=zoom_level, b=0, c=0, d=zoom_level,
            tx=screen_width/2 - camera_x * zoom_level,
            ty=screen_height/2 - camera_y * zoom_level
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
        help_text = font.render("Mouse wheel: Zoom, R: Reset, +/-: Zoom", True, (255, 255, 255))
        screen.blit(zoom_text, (10, 10))
        screen.blit(help_text, (10, 50))
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
