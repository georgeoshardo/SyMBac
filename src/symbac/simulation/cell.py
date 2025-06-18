# cell.py (Updated)
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np
from typing import Optional
from symbac.misc import generate_color
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class CellConfig:
    GRANULARITY: int = 8  # Number of segments per cell radius
    SEGMENT_RADIUS: float = 15.0
    SEGMENT_MASS: float = 1.0
    GROWTH_RATE: float = 5.0
    MIN_LENGTH_AFTER_DIVISION: int = 10
    MAX_LENGTH_VARIATION: float = 0.2
    BASE_MAX_LENGTH: int = 40
    SEED_CELL_SEGMENTS: int = 15
    MAX_BEND_ANGLE: float = 0.05  # 0.01 normally, 0.05 also good for E. coli in MM
    STIFFNESS: int = 300_000

    DAMPED_ROTARY_SPRING: bool = False
    ROTARY_SPRING_STIFFNESS: float | None = None
    ROTARY_SPRING_DAMPING:  float | None = None

    def __post_init__(self):
        if not self.DAMPED_ROTARY_SPRING:
            if self.ROTARY_SPRING_STIFFNESS is not None or self.ROTARY_SPRING_DAMPING is not None:
                raise ValueError(
                    "Cannot set ROTARY_SPRING_STIFFNESS or ROTARY_SPRING_DAMPING "
                    "when ROTARY_SPRING is False."
                )
        else:
            if self.ROTARY_SPRING_STIFFNESS is None:
                raise ValueError(
                    "DAMPED_ROTARY_SPRING=True, but ROTARY_SPRING_STIFFNESS was not provided."
                )
            if self.ROTARY_SPRING_DAMPING is None:
                raise ValueError(
                    "DAMPED_ROTARY_SPRING=True, but ROTARY_SPRING_DAMPING was not provided."
                )






# Note that length units here are the number of spheres in the cell, TODO: implement the continuous length measurement for rendering.
class Cell:
    def __init__(
            self,
            space: pymunk.Space,
            config: CellConfig,
            start_pos: tuple[float, float],
            group_id: int = 0,
            noise_strength: float = 0.05,
            _from_division: bool = False
    ) -> None:
        """
        Initialize a segmented (bendy) cell instance.

        Parameters
        ----------
        space : pymunk.Space
            The simulation space where the cell exists and interacts with other
            physical entities.
        start_pos : tuple[float, float]
            The starting position (x, y) for creating the first segment of the cell.
        num_segments : int
            The initial number of segments to create for this cell.
        segment_radius : float
            The radius of each segment in the cell.
        segment_mass : float
            The mass of each segment in the cell.
        group_id : int, optional
            A unique identifier for grouping or categorizing the segments in cell.
            TODO: use for lineage tracking, give daughter the mother ID
        growth_rate : float, optional
            The rate at which the cell grows over time.
        min_length_after_division : int, optional
            The minimal length the cell must have after it undergoes division. Defaults to 10.
        max_length_variation : float, optional
            The percentage variation allowed in determining the maximum length of the cell.
        noise_strength : float, optional
            The strength of random noisy forces added to modulate the cell's dynamics.
        base_max_length : int, optional
            An optional base value for calculating the randomised maximum length. If not
            provided, the value of max_length is used.
        _from_division : bool, optional
            Indicates whether the cell is being created as a result of division.
        """
        self.space = space
        self.config = config
        self.start_pos = start_pos
        self.noise_strength = noise_strength

        self.group_id = group_id
        self.base_color = generate_color(group_id)

        # Rest of the existing code...
        self.bodies = []
        self.shapes = []
        self.joints = []

        self.growth_accumulator = 0.0
        self.growth_threshold = self.config.SEGMENT_RADIUS / self.config.GRANULARITY
        self.joint_distance = self.config.SEGMENT_RADIUS / self.config.GRANULARITY

        # --- TODO: ADD/MODIFY THESE ATTRIBUTES, make them controllable? ---
        self.is_dividing = False
        self.septum_progress = 0.0
        self.septum_duration = 1.5  # Duration of septum formation in seconds
        self.division_site = None
        self.num_septum_segments = self.config.GRANULARITY  # HOW MANY segments on each side form the septum
        self.min_septum_radius = 0.1  # HOW SMALL the center of the septum gets
        self.length_at_division_start = 0
        self.division_bias = self.config.GRANULARITY  # Segment-level division bias

        # Always randomise from the original base, not the parent's randomised value
        variation = self.config.BASE_MAX_LENGTH * self.config.MAX_LENGTH_VARIATION
        random_max_len = np.random.uniform(
            self.config.BASE_MAX_LENGTH - variation, self.config.BASE_MAX_LENGTH + variation
            # TODO what is the true distribution of division lengths?
        )

        self._max_length = max(self.config.MIN_LENGTH_AFTER_DIVISION * 2, int(random_max_len))
        # print(self.config.BASE_MAX_LENGTH, random_max_len, self._max_length)

        if not _from_division:  # Seed cell
            for i in range(self.config.SEED_CELL_SEGMENTS):
                self._add_initial_segment(i == 0)
            self._update_colors()

    def _add_initial_segment(self, is_first: bool) -> None:

        """
            Adds a single segment to the cell during initialization.
            """
        moment = pymunk.moment_for_circle(
            self.config.SEGMENT_MASS, 0, self.config.SEGMENT_RADIUS
        )
        body = pymunk.Body(self.config.SEGMENT_MASS, moment)

        if is_first:
            body.position = self.start_pos
        else:
            prev_body = self.bodies[-1]
            # Keep growth perfectly straight
            offset = Vec2d(self.joint_distance, 0).rotated(prev_body.angle)
            body.position = prev_body.position + offset
            body.angle = prev_body.angle
            # Add tiny random positional noise to break determinism
            noise_x = np.random.uniform(-0.1, 0.1)
            noise_y = np.random.uniform(-0.1, 0.1)
            body.position += Vec2d(noise_x, noise_y)

        shape = pymunk.Circle(body, self.config.SEGMENT_RADIUS)
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
            pivot.max_force = self.config.STIFFNESS
            self.space.add(pivot)
            self.joints.append(pivot)

            limit = pymunk.RotaryLimitJoint(
                prev_body, body, -self.config.MAX_BEND_ANGLE, self.config.MAX_BEND_ANGLE
            )
            limit.max_force = self.config.STIFFNESS
            self.space.add(limit)
            self.joints.append(limit)

            if self.config.DAMPED_ROTARY_SPRING:
                # --- Add a DampedRotarySpring for internal stability ---
                assert self.config.ROTARY_SPRING_STIFFNESS is not None # For mypy type checking
                assert self.config.ROTARY_SPRING_DAMPING is not None # For mypy type checking
                spring = pymunk.DampedRotarySpring(
                    prev_body, body, 0, self.config.ROTARY_SPRING_STIFFNESS, self.config.ROTARY_SPRING_DAMPING
                )
                self.space.add(spring)
                self.joints.append(spring)

    def apply_noise(self, dt: float) -> None:

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
        Growth continues during division.
        """
        # Allow growth to continue if the cell is dividing, otherwise stop at max length
        if not self.is_dividing and len(self.bodies) >= self._max_length:
            return

        if len(self.bodies) < 2:
            return

        # User change: randomized growth
        self.growth_accumulator += (
                self.config.GROWTH_RATE * dt * np.random.uniform(0, 4)
        )

        # There are 3 joints per connection now, so we get the pivot from the end
        print(self.joints)
        last_pivot_joint = self.joints[-3]
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
                self.config.SEGMENT_MASS, 0, self.config.SEGMENT_RADIUS
            )
            new_tail_body = pymunk.Body(self.config.SEGMENT_MASS, moment)

            # Keep growth direction perfectly straight
            new_tail_offset = Vec2d(self.joint_distance, 0).rotated(
                old_tail_body.angle
            )
            new_tail_body.position = old_tail_body.position + new_tail_offset
            new_tail_body.angle = old_tail_body.angle

            # NEW: Add tiny random positional noise to the new segment
            noise_x = np.random.uniform(-0.1, 0.1)
            noise_y = np.random.uniform(-0.1, 0.1)
            new_tail_body.position += Vec2d(noise_x, noise_y)

            new_tail_shape = pymunk.Circle(new_tail_body, self.config.SEGMENT_RADIUS)
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
            new_pivot.max_force = self.config.STIFFNESS
            self.space.add(new_pivot)
            self.joints.append(new_pivot)

            new_limit = pymunk.RotaryLimitJoint(
                old_tail_body,
                new_tail_body,
                -self.config.MAX_BEND_ANGLE,
                self.config.MAX_BEND_ANGLE,
            )
            new_limit.max_force = self.config.STIFFNESS
            self.space.add(new_limit)
            self.joints.append(new_limit)

            if self.config.DAMPED_ROTARY_SPRING:
                # --- Add a DampedRotarySpring for internal stability ---
                new_spring = pymunk.DampedRotarySpring(
                    old_tail_body, new_tail_body, 0, self.config.ROTARY_SPRING_STIFFNESS, self.config.ROTARY_SPRING_DAMPING
                )
                self.space.add(new_spring)
                self.joints.append(new_spring)

            self.growth_accumulator = 0.0
            self._update_colors()

    # ... The rest of your Cell class methods (_split_cell, divide, etc.) remain unchanged ...
    # ... I have omitted them for brevity but they are still part of the class.
    # Make sure to adjust the indices for joint removal during division (_split_cell).

    def divide(self, next_group_id: int, dt: float) -> Optional['Cell']:
        """
        Manages division with a multi-segment, tapered septum formation.
        This version correctly handles continued growth to ensure a symmetric split.
        """
        # 1. INITIATE DIVISION
        if not self.is_dividing:
            if len(self.bodies) < self._max_length:
                return None

            split_index = len(self.bodies) // 2
            if split_index < self.config.MIN_LENGTH_AFTER_DIVISION or \
                    (len(self.bodies) - split_index) < self.config.MIN_LENGTH_AFTER_DIVISION:
                return None

            # Start the division process
            self.is_dividing = True
            self.septum_progress = 0.0
            self.division_site = split_index
            self.length_at_division_start = len(self.bodies)  # Record initial length
            return None

        # 2. CONTINUE AND UPDATE SEPTUM / SPLIT IF READY
        if self.is_dividing:
            self.septum_progress += dt / self.septum_duration

            # Perform the split logic (which includes septum drawing and final division)
            daughter_cell = self._split_cell(next_group_id)

            # If a daughter cell was returned, it means division is complete
            if daughter_cell:
                # Reset the mother cell's state
                self.is_dividing = False
                self.septum_progress = 0.0
                self.division_site = None
                self.length_at_division_start = 0

            return daughter_cell

        return None

    def remove_tail_segment(self):
        """
        Safely removes the last segment of the cell.
        """
        if len(self.bodies) <= self.config.MIN_LENGTH_AFTER_DIVISION:
            return

        tail_body = self.bodies.pop()
        tail_shape = self.shapes.pop()

        # --- UPDATED: Remove 3 joints for the last segment ---
        for _ in range(3):
            if self.joints:
                joint_to_remove = self.joints.pop()
                self.space.remove(joint_to_remove)

        self.space.remove(tail_body, tail_shape)
        self._update_colors()

    def _update_colors(self):
        """
        Sets the head and tail colors based on the cell's base color.
        """
        if not self.shapes:
            return

        # NEW: Use the cell's base color with variations
        r, g, b = self.base_color
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

    def _split_cell(self, next_group_id: int) -> Optional['Cell']:
        """
        Splits the current cell into two distinct cells if septum formation is complete.
        """
        progress = min(1.0, self.septum_progress)

        # --- Draw the septum on the segments around the original division site ---
        for i in range(self.num_septum_segments):
            mother_idx = self.division_site - 1 - i
            daughter_idx = self.division_site + i

            if mother_idx >= 0 and daughter_idx < len(self.shapes):
                falloff = (self.num_septum_segments - i) / self.num_septum_segments
                shrinkage = (self.config.SEGMENT_RADIUS - self.min_septum_radius) * progress * falloff
                new_radius = self.config.SEGMENT_RADIUS - shrinkage

                old_mother_shape = self.shapes[mother_idx]
                self.shapes[mother_idx] = self._recreate_shape(old_mother_shape, new_radius)

                old_daughter_shape = self.shapes[daughter_idx]
                self.shapes[daughter_idx] = self._recreate_shape(old_daughter_shape, new_radius)

        if progress < 1.0:
            return None

        for i in range(self.num_septum_segments):
            mother_idx = self.division_site - 1 - i
            daughter_idx = self.division_site + i
            if mother_idx >= 0 and daughter_idx < len(self.shapes):
                self.shapes[mother_idx] = self._recreate_shape(self.shapes[mother_idx], self.config.SEGMENT_RADIUS)
                self.shapes[daughter_idx] = self._recreate_shape(self.shapes[daughter_idx], self.config.SEGMENT_RADIUS)

        current_length = len(self.bodies)
        growth_during_division = current_length - self.length_at_division_start
        original_half_length = self.length_at_division_start // 2
        growth_to_redistribute_to_mother = growth_during_division // 2
        mother_final_len = original_half_length + growth_to_redistribute_to_mother
        mother_final_len += self.division_bias
        daughter_length = current_length - mother_final_len

        if mother_final_len < self.config.MIN_LENGTH_AFTER_DIVISION:
            mother_final_len = self.config.MIN_LENGTH_AFTER_DIVISION
        elif daughter_length < self.config.MIN_LENGTH_AFTER_DIVISION:
            mother_final_len = current_length - self.config.MIN_LENGTH_AFTER_DIVISION
        elif mother_final_len >= current_length:
            mother_final_len = current_length - self.config.MIN_LENGTH_AFTER_DIVISION

        daughter_cell = Cell(
            space=self.space,
            config=self.config,
            start_pos=self.bodies[mother_final_len].position,
            group_id=next_group_id,
            noise_strength=self.noise_strength,
            _from_division=True)

        # --- UPDATED: Joint indexing now accounts for 3 joints per connection ---
        daughter_cell.bodies = self.bodies[mother_final_len:]
        daughter_cell.shapes = self.shapes[mother_final_len:]
        daughter_cell.joints = self.joints[mother_final_len * 3:]

        for shape in daughter_cell.shapes:
            shape.filter = pymunk.ShapeFilter(group=next_group_id)

        connecting_joint_index = (mother_final_len - 1) * 3
        if connecting_joint_index < len(self.joints):
            # Remove all three connecting joints
            for i in range(3):
                joint_to_remove = self.joints[connecting_joint_index + i]
                self.space.remove(joint_to_remove)

        self.bodies = self.bodies[:mother_final_len]
        self.shapes = self.shapes[:mother_final_len]
        self.joints = self.joints[:connecting_joint_index]

        self._update_colors()
        daughter_cell._update_colors()

        return daughter_cell

    def _recreate_shape(self, shape_to_replace: pymunk.Circle, new_radius: float) -> pymunk.Circle:
        """
        Recreates a shape by replacing it with a new shape of a different radius.
        """
        body = shape_to_replace.body
        friction = shape_to_replace.friction
        filter = shape_to_replace.filter
        color = shape_to_replace.color
        self.space.remove(shape_to_replace)
        new_shape = pymunk.Circle(body, new_radius)
        new_shape.friction = friction
        new_shape.filter = filter
        new_shape.color = color
        self.space.add(new_shape)
        return new_shape