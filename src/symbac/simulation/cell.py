import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np
from typing import Optional
from symbac.misc import generate_color

#Note that length units here are the number of spheres in the cell, TODO: implement the continuous length measurement for rendering.
class Cell:
    space: pymunk.Space
    start_pos: tuple[float, float]
    num_segments: int
    segment_radius: float
    segment_mass: float
    group_id: int
    growth_rate: float
    min_length_after_division: int
    max_length_variation: float
    base_max_length: Optional[int]
    _from_division: Optional[bool]

    max_bend_angle: float
    base_min_length_after_division: int
    base_max_length_variation: float
    noise_strength: float
    _max_length: int
    base_color: tuple[int, int, int]


    def __init__(
            self,
            space: pymunk.Space,
            start_pos: tuple[float, float],
            num_segments: int,
            segment_radius: float,
            segment_mass: float,
            group_id: int = 0,
            growth_rate: float = 5.0,
            min_length_after_division: int = 10,
            max_length_variation: float = 0.2,
            noise_strength: float = 0.05,
            base_max_length: int = 40,
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
        _max_length : int, optional
            The maximum allowable length of the cell in terms of the number of segments.
        min_length_after_division : int, optional
            The minimal length the cell must have after it undergoes division. Defaults to 10.
        max_length_variation : float, optional
            The percentage variation allowed in determining the maximum length of the cell.
        base_color : tuple[int, int, int], optional
            The base color of the cell when displayed in Pygame, specified as an RGB tuple. If None, a color is
            generated based on the group_id.
            TODO: switch to Pyglet
        noise_strength : float, optional
            The strength of random noisy forces added to modulate the cell's dynamics.
        base_max_length : int, optional
            An optional base value for calculating the randomised maximum length. If not
            provided, the value of max_length is used.
        _from_division : bool, optional
            Indicates whether the cell is being created as a result of division.
        """
        self.space = space
        self.start_pos = start_pos
        self.segment_radius = segment_radius
        self.segment_mass = segment_mass
        self.growth_rate = growth_rate
        self.max_bend_angle = 0.005  # 0.01 normally
        self.noise_strength = noise_strength

        self.group_id = group_id
        self.base_color = generate_color(group_id)

        # Store the original base max_length for consistent inheritance
        self.base_max_length = base_max_length

        # Always randomise from the original base, not the parent's randomised value
        variation = self.base_max_length * max_length_variation
        random_max_len = np.random.uniform(
            self.base_max_length - variation, self.base_max_length + variation
        )
        self._max_length = max(min_length_after_division * 2, int(random_max_len))

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

        # --- ADD THESE NEW ATTRIBUTES ---
        self.is_dividing = False
        self.septum_progress = 0.0
        self.septum_duration = 1.5  # Duration of septum formation in seconds
        self.division_site = None
        self.num_septum_segments = 4  # HOW MANY segments on each side form the septum
        self.min_septum_radius = 0.1  # HOW SMALL the center of the septum gets
        # --- END OF NEW ATTRIBUTES ---


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
        Growth continues during division.
        """
        # Allow growth to continue if the cell is dividing, otherwise stop at max length
        if not self.is_dividing and len(self.bodies) >= self._max_length:
            return

        if len(self.bodies) < 2:
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

        # In cell.py, replace the old divide method with this one

        # In cell.py, replace the divide method with this corrected version

        # In cell.py, replace the entire divide method with this one

    def divide(self, next_group_id: int, dt: float) -> Optional['Cell']:
        """
        Manages division with a multi-segment, tapered septum formation.
        """
        # 1. INITIATE DIVISION
        if not self.is_dividing:
            if len(self.bodies) < self._max_length:
                return None

            split_index = len(self.bodies) // 2
            if split_index < self.min_length_after_division or \
                    (len(self.bodies) - split_index) < self.min_length_after_division:
                return None

            self.is_dividing = True
            self.septum_progress = 0.0
            self.division_site = split_index
            return None

        # 2. CONTINUE AND UPDATE TAPERED SEPTUM
        if self.is_dividing:
            self.septum_progress += dt / self.septum_duration
            progress = min(1.0, self.septum_progress)

            # Loop through the segments that form the septum
            for i in range(self.num_septum_segments):
                # Determine the indices of the segments on the mother and daughter sides
                mother_idx = self.division_site - 1 - i
                daughter_idx = self.division_site + i

                # Ensure the indices are within the bounds of the cell's body
                if mother_idx < 0 or daughter_idx >= len(self.bodies):
                    continue

                # Create a falloff effect: segments closer to the center (i=0) shrink more
                falloff = (self.num_septum_segments - i) / self.num_septum_segments

                # Calculate the amount of shrinkage based on progress and falloff
                shrinkage = (self.segment_radius - self.min_septum_radius) * progress * falloff
                new_radius = self.segment_radius - shrinkage

                if progress < 1.0:
                    # Recreate both shapes with their new, smaller radius
                    old_mother_shape = self.shapes[mother_idx]
                    self.shapes[mother_idx] = self._recreate_shape(old_mother_shape, new_radius)

                    old_daughter_shape = self.shapes[daughter_idx]
                    self.shapes[daughter_idx] = self._recreate_shape(old_daughter_shape, new_radius)

            # 3. COMPLETE DIVISION
            if progress >= 1.0:
                # Restore all affected segments to their original size before splitting
                for i in range(self.num_septum_segments):
                    mother_idx = self.division_site - 1 - i
                    daughter_idx = self.division_site + i
                    if mother_idx < 0 or daughter_idx >= len(self.shapes):
                        continue

                    # Restore mother segment
                    old_mother_shape = self.shapes[mother_idx]
                    if old_mother_shape.radius < self.segment_radius:
                        self.shapes[mother_idx] = self._recreate_shape(old_mother_shape, self.segment_radius)

                    # Restore daughter segment
                    old_daughter_shape = self.shapes[daughter_idx]
                    if old_daughter_shape.radius < self.segment_radius:
                        self.shapes[daughter_idx] = self._recreate_shape(old_daughter_shape, self.segment_radius)

                # Perform the actual split
                daughter_cell = self._split_cell(next_group_id)

                # Reset the mother cell's state
                self.is_dividing = False
                self.septum_progress = 0.0
                self.division_site = None

                return daughter_cell

        return None

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


# In cell.py, add this new private method to the Cell class

    def _split_cell(self, next_group_id: int) -> Optional['Cell']:
        """
        Performs the actual separation of the cell after the septum has formed.
        This contains the logic from the original divide method.
        """
        # Create daughter cell - pass the BASE max_length, not this cell's randomized one
        daughter_cell = Cell(
            space=self.space,
            start_pos=self.bodies[self.division_site].position,
            num_segments=0,  # Start with 0 and add them manually
            segment_radius=self.segment_radius,
            segment_mass=self.segment_mass,
            group_id=next_group_id,
            growth_rate=self.growth_rate,
            base_max_length=self.base_max_length,
            min_length_after_division=self.min_length_after_division,
            max_length_variation=self.max_length_variation,
            noise_strength=self.noise_strength,
            _from_division=True)

        # Partition the mother's parts.
        daughter_cell.bodies = self.bodies[self.division_site:]
        daughter_cell.shapes = self.shapes[self.division_site:]
        daughter_cell.joints = self.joints[self.division_site * 2:] # There are 2 joints per segment

        for shape in daughter_cell.shapes:
            shape.filter = pymunk.ShapeFilter(group=next_group_id)

        # Remove the single joint connecting the two new cells
        connecting_joint_index = (self.division_site - 1) * 2
        connecting_joint = self.joints[connecting_joint_index]
        connecting_limit = self.joints[connecting_joint_index + 1]
        self.space.remove(connecting_joint, connecting_limit)

        # Trim the mother cell
        self.bodies = self.bodies[:self.division_site]
        self.shapes = self.shapes[:self.division_site]
        self.joints = self.joints[:connecting_joint_index]

        self._update_colors()
        daughter_cell._update_colors()

        return daughter_cell

# In cell.py, add this new private method to the Cell class

    def _recreate_shape(self, shape_to_replace, new_radius):
        """
        Removes a shape, creates a new one with a new radius attached to the
        same body, and adds it back to the space. Preserves properties.
        Returns the new shape.
        """
        body = shape_to_replace.body

        # Store the properties of the old shape
        friction = shape_to_replace.friction
        filter = shape_to_replace.filter
        color = shape_to_replace.color

        # Remove the old shape from the space and the cell's list
        self.space.remove(shape_to_replace)

        # Create a new shape with the new radius
        new_shape = pymunk.Circle(body, new_radius)
        new_shape.friction = friction
        new_shape.filter = filter
        new_shape.color = color

        # Add the new shape to the space
        self.space.add(new_shape)

        return new_shape