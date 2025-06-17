import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np
from typing import Optional

def generate_color(group_id) -> tuple[int, int, int]:
    """
    Generate a unique color based on group_id using HSV color space
    for better visual distinction between cells.
    """
    import colorsys

    # Use golden ratio for better color distribution
    golden_ratio = 0.618033988749895
    hue = (group_id * golden_ratio) % 1.0
    saturation = 0.7 + (group_id % 3) * 0.1  # Vary saturation slightly
    value = 0.8 + (group_id % 2) * 0.2  # Vary brightness slightly

    rgb: tuple[float, float, float] = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)


#Note that length units here are the number of spheres in the cell, TODO: implement the continuous length measurement for rendering.
class Cell:
    space: pymunk.Space
    start_pos: tuple[float, float]
    num_segments: int
    segment_radius: float
    segment_mass: float
    group_id: int
    growth_rate: float
    max_length: int
    min_length_after_division: int
    max_length_variation: float
    base_color: Optional[tuple[int, int, int]]
    base_max_length: Optional[int]
    _from_division: Optional[bool]

    max_bend_angle: float
    base_min_length_after_division: int
    base_max_length_variation: float
    noise_strength: float

    def __init__(
            self,
            space: pymunk.Space,
            start_pos: tuple[float, float],
            num_segments: int,
            segment_radius: float,
            segment_mass: float,
            group_id: int = 0,
            growth_rate: float = 5.0,
            max_length: int = 40,
            min_length_after_division: int = 10,
            max_length_variation: float = 0.2,
            base_color: Optional[tuple[int, int, int]] = None,
            noise_strength: float = 0.05,
            base_max_length: Optional[int] = None,
            _from_division: bool = False
    ) -> None:

        self.space = space
        self.start_pos = start_pos
        self.segment_radius = segment_radius
        self.segment_mass = segment_mass
        self.growth_rate = growth_rate
        self.max_bend_angle = 0.005  # 0.01 normally
        self.noise_strength = noise_strength

        self.group_id = group_id
        self.base_color = base_color if base_color else generate_color(group_id)

        # NEW: Store the original base max_length for consistent inheritance
        self.base_max_length = base_max_length if base_max_length is not None else max_length

        # Always randomize from the original base, not the parent's randomized value
        variation = self.base_max_length * max_length_variation
        random_max_len = np.random.uniform(
            self.base_max_length - variation, self.base_max_length + variation
        )
        self.max_length = max(min_length_after_division * 2, int(random_max_len))

        self.min_length_after_division = min_length_after_division
        self.max_length_variation = max_length_variation

        # Rest of the existing code...
        self.bodies = []
        self.shapes = []
        self.joints = []

        self.growth_accumulator = 0.0
        self.growth_threshold = self.segment_radius / 3
        self.joint_distance = self.segment_radius / 4
        self.joint_max_force = 30000

        if not _from_division:
            for i in range(num_segments):
                self._add_initial_segment(i == 0)
            self._update_colors()

    def _add_initial_segment(self, is_first):
        """
        Adds a single segment to the cell during initialization.
        """
        moment = pymunk.moment_for_circle(
            self.segment_mass, 0, self.segment_radius
        )
        body = pymunk.Body(self.segment_mass, moment)

        if is_first:
            body.position = self.start_pos
        else:
            prev_body = self.bodies[-1]
            # Keep growth perfectly straight
            offset = Vec2d(self.joint_distance, 0).rotated(prev_body.angle)
            body.position = prev_body.position + offset

            # Add tiny random positional noise to break determinism
            noise_x = np.random.uniform(-0.1, 0.1)
            noise_y = np.random.uniform(-0.1, 0.1)
            body.position += Vec2d(noise_x, noise_y)

        shape = pymunk.Circle(body, self.segment_radius)
        shape.friction = 0.0
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

    def apply_noise(self, dt):
        """
        NEW: Apply small random forces to all segments to simulate environmental noise
        """
        for body in self.bodies:
            # Apply tiny random forces
            force_x = np.random.uniform(-self.noise_strength, self.noise_strength)
            force_y = np.random.uniform(-self.noise_strength, self.noise_strength)
            body.force += Vec2d(force_x, force_y)

            # Also apply tiny random torques
            torque = np.random.uniform(-self.noise_strength * 0.1, self.noise_strength * 0.1)
            body.torque += torque

    def grow(self, dt):
        """
        Grows the cell by extending the last segment until a new one can be added.
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

            # Keep growth direction perfectly straight
            new_tail_offset = Vec2d(self.joint_distance, 0).rotated(
                old_tail_body.angle
            )
            new_tail_body.position = old_tail_body.position + new_tail_offset

            # NEW: Add tiny random positional noise to the new segment
            noise_x = np.random.uniform(-0.1, 0.1)
            noise_y = np.random.uniform(-0.1, 0.1)
            new_tail_body.position += Vec2d(noise_x, noise_y)

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

    def divide(self, next_group_id: int) -> Optional['Cell']:

        """
            If the cell is at max_length, it splits by transplanting its second
            half into a new Cell object, preserving orientation.
            """
        if len(self.bodies) < self.max_length:
            return None

        split_index = len(self.bodies) // 2
        if split_index < self.min_length_after_division or (
                len(self.bodies) - split_index
        ) < self.min_length_after_division:
            return None

        # Create daughter cell - pass the BASE max_length, not this cell's randomized one
        daughter_cell = Cell(
            self.space,
            self.bodies[split_index].position,
            0,
            self.segment_radius,
            self.segment_mass,
            next_group_id,
            self.growth_rate,
            self.base_max_length,  # FIXED: Pass the original base length
            self.min_length_after_division,
            self.max_length_variation,
            base_color=generate_color(next_group_id),
            noise_strength=self.noise_strength,
            base_max_length=self.base_max_length,  # NEW: Ensure base is preserved
            _from_division=True,
        )

        # Partition the mother's parts.
        daughter_cell.bodies = self.bodies[split_index:]
        daughter_cell.shapes = self.shapes[split_index:]
        daughter_cell.joints = self.joints[split_index * 2:]

        for shape in daughter_cell.shapes:
            shape.filter = pymunk.ShapeFilter(group=next_group_id)

        connecting_joint = self.joints[(split_index - 1) * 2]
        connecting_limit = self.joints[(split_index - 1) * 2 + 1]
        self.space.remove(connecting_joint, connecting_limit)

        self.bodies = self.bodies[:split_index]
        self.shapes = self.shapes[:split_index]
        self.joints = self.joints[: (split_index - 1) * 2]

        self._update_colors()
        daughter_cell._update_colors()

        return daughter_cell

    def remove_tail_segment(self):
        """
        Safely removes the last segment of the cell.
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
        """
        Sets the head and tail colors based on the cell's base color.
        """
        if not self.shapes:
            return

        # NEW: Use the cell's base color with variations
        r, g, b = self.base_color

        # Body segments: use base color with alpha
        body_color = (r, g, b, 255)

        # Head: brighter version of base color
        head_color = (
            min(255, int(r * 1.3)),
            min(255, int(g * 1.3)),
            min(255, int(b * 1.3)),
            255
        )

        # Tail: darker version of base color
        tail_color = (
            int(r * 0.7),
            int(g * 0.7),
            int(b * 0.7),
            255
        )

        # Apply colors
        for shape in self.shapes:
            shape.color = body_color

        self.shapes[0].color = head_color  # Head
        self.shapes[-1].color = tail_color  # Tail
