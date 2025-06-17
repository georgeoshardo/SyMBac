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
        self.start_pos = start_pos
        self.segment_radius = segment_radius
        self.segment_mass = segment_mass
        self.growth_rate = growth_rate
        self.max_bend_angle = 0.005  # 0.01 normally, 0.05 also good for E. coli in MM
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
        self.joint_max_force = 300000 # Increase this a lot to make cells rigid, and reduce the max bend angle to 0 (may cause instability)

        # --- TODO: ADD/MODIFY THESE ATTRIBUTES, make them controllable? ---
        self.is_dividing = False
        self.septum_progress = 0.0
        self.septum_duration = 1.5  # Duration of septum formation in seconds
        self.division_site = None
        self.num_septum_segments = 4  # HOW MANY segments on each side form the septum
        self.min_septum_radius = 0.1  # HOW SMALL the center of the septum gets
        self.length_at_division_start = 0
        self.division_bias = 3 # Segment-level division bias


        if not _from_division:
            for i in range(num_segments):
                self._add_initial_segment(i == 0)
            self._update_colors()

    def _add_initial_segment(self, is_first: bool) -> None:

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
            body.angle = prev_body.angle
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
            new_tail_body.angle = old_tail_body.angle

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
            if split_index < self.min_length_after_division or \
                    (len(self.bodies) - split_index) < self.min_length_after_division:
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

    def _split_cell(self, next_group_id: int) -> Optional['Cell']:
        """
        Splits the current cell into two distinct cells if septum formation is complete.

        This method manages the process of cell division by simulating the formation and
        pinching of the septum, which determines the separation boundary of the two cells.
        It modifies the shapes and other attributes of the current cell to accommodate
        the progression of division. Once the septum is fully formed, the cell is split
        into a mother cell and a daughter cell. The daughter cell inherits certain
        attributes and state from the mother cell. Cell division is symmetric with the
        possibility of a bias applied to distribute extra segments between mother and
        daughter cells.

        Parameters
        ----------
        next_group_id : int
            The group ID to assign to the newly created daughter cell.

        Returns
        -------
        Optional[Cell]
            The daughter cell created after the split, if division is complete.
            Returns `None` if division has not yet completed its process.
        """
        progress = min(1.0, self.septum_progress)

        # --- Draw the septum on the segments around the original division site ---
        for i in range(self.num_septum_segments):
            mother_idx = self.division_site - 1 - i
            daughter_idx = self.division_site + i

            if mother_idx >= 0 and daughter_idx < len(self.shapes):
                falloff = (self.num_septum_segments - i) / self.num_septum_segments
                shrinkage = (self.segment_radius - self.min_septum_radius) * progress * falloff
                new_radius = self.segment_radius - shrinkage

                # Recreate shapes to apply the new radius
                old_mother_shape = self.shapes[mother_idx]
                self.shapes[mother_idx] = self._recreate_shape(old_mother_shape, new_radius)

                old_daughter_shape = self.shapes[daughter_idx]
                self.shapes[daughter_idx] = self._recreate_shape(old_daughter_shape, new_radius)

        # --- Check if it's time to complete the division ---
        if progress < 1.0:
            return None  # Not done pinching yet, return nothing.

        # --- Time to split: Restore radii and perform a symmetric partition ---
        # First, restore the radii of the pinched segments back to normal
        for i in range(self.num_septum_segments):
            mother_idx = self.division_site - 1 - i
            daughter_idx = self.division_site + i
            if mother_idx >= 0 and daughter_idx < len(self.shapes):
                self.shapes[mother_idx] = self._recreate_shape(self.shapes[mother_idx], self.segment_radius)
                self.shapes[daughter_idx] = self._recreate_shape(self.shapes[daughter_idx], self.segment_radius)

        # Calculate symmetric division accounting for growth during division
        current_length = len(self.bodies)
        growth_during_division = current_length - self.length_at_division_start

        # The original division site was at length_at_division_start // 2
        # Growth occurred at the tail, so we need to give half of that growth to the mother
        original_half_length = self.length_at_division_start // 2
        growth_to_redistribute_to_mother = growth_during_division // 2

        # New symmetric split point: original half + half the growth that occurred
        mother_final_len = original_half_length + growth_to_redistribute_to_mother

        # NEW: Apply division bias (positive = more segments to mother, negative = more to daughter)
        mother_final_len += self.division_bias

        # Ensure both cells meet minimum length requirements
        daughter_length = current_length - mother_final_len
        if mother_final_len < self.min_length_after_division:
            mother_final_len = self.min_length_after_division
        elif daughter_length < self.min_length_after_division:
            mother_final_len = current_length - self.min_length_after_division
        elif mother_final_len >= current_length:
            mother_final_len = current_length - self.min_length_after_division

        # Create daughter cell with inherited bias
        daughter_cell = Cell(
            space=self.space,
            start_pos=self.bodies[mother_final_len].position,
            num_segments=0,
            segment_radius=self.segment_radius,
            segment_mass=self.segment_mass,
            group_id=next_group_id,
            growth_rate=self.growth_rate,
            base_max_length=self.base_max_length,
            min_length_after_division=self.min_length_after_division,
            max_length_variation=self.max_length_variation,
            noise_strength=self.noise_strength,
            _from_division=True)

        # Partition the cell's components based on the adjusted split point
        daughter_cell.bodies = self.bodies[mother_final_len:]
        daughter_cell.shapes = self.shapes[mother_final_len:]
        daughter_cell.joints = self.joints[mother_final_len * 2:]

        for shape in daughter_cell.shapes:
            shape.filter = pymunk.ShapeFilter(group=next_group_id)

        # Remove the connecting joint between mother and daughter
        connecting_joint_index = (mother_final_len - 1) * 2
        if connecting_joint_index < len(self.joints) and connecting_joint_index + 1 < len(self.joints):
            connecting_joint = self.joints[connecting_joint_index]
            connecting_limit = self.joints[connecting_joint_index + 1]
            self.space.remove(connecting_joint, connecting_limit)

        # Trim the mother cell to its final, correct length
        self.bodies = self.bodies[:mother_final_len]
        self.shapes = self.shapes[:mother_final_len]
        self.joints = self.joints[:connecting_joint_index]

        self._update_colors()
        daughter_cell._update_colors()

        return daughter_cell

    def _recreate_shape(self, shape_to_replace: pymunk.Circle, new_radius: float) -> pymunk.Circle:
        """
        Recreates a shape by replacing it with a new shape of a different radius,
        while maintaining the original shape's properties, such as friction, filter,
        and color. The old shape is removed from the space, and a new one is added.

        Parameters
        ----------
        shape_to_replace : pymunk.Circle
            The original shape to be replaced. Its friction, filter, and color
            properties will be transferred to the newly created shape.
        new_radius : float
            The radius of the new shape that will replace the old shape.

        Returns
        -------
        pymunk.Circle
            The newly created shape with the provided radius, added to the space, and
            retaining the original shape's properties.
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