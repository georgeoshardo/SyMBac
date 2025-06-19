import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np
from typing import Optional
from symbac.misc import generate_color
from symbac.simulation.config import CellConfig
from symbac.simulation.segments import CellSegment

# Note that length units here are the number of spheres in the cell, TODO: implement the continuous length measurement for rendering.
class Cell:
    def __init__(
            self,
            space: pymunk.Space,
            config: CellConfig,
            start_pos: tuple[float, float],
            group_id: int = 0,
            _from_division: bool = False
    ) -> None:
        self.space = space
        self.config = config
        self.start_pos = start_pos
        self.group_id = group_id
        self.base_color = generate_color(group_id)

        self.bodies: list[pymunk.Body] = []
        self.shapes: list[pymunk.Circle] = []

        self.pivot_joints: list[pymunk.PivotJoint] = []
        self.limit_joints: list[pymunk.RotaryLimitJoint] = []
        self.spring_joints: list[pymunk.DampedRotarySpring] = []

        self.growth_accumulator = 0.0
        self.growth_threshold = self.config.SEGMENT_RADIUS / self.config.GRANULARITY
        self.joint_distance = self.config.SEGMENT_RADIUS / self.config.GRANULARITY

        self.is_dividing = False
        self.septum_progress = 0.0
        self.septum_duration = 1.5
        self.division_site: int | None = None
        self.num_septum_segments = self.config.GRANULARITY
        self.min_septum_radius = 0.1
        self.length_at_division_start = 0
        self.division_bias = self.config.GRANULARITY

        variation = self.config.BASE_MAX_LENGTH * self.config.MAX_LENGTH_VARIATION
        random_max_len = np.random.uniform(
            self.config.BASE_MAX_LENGTH - variation, self.config.BASE_MAX_LENGTH + variation
        )

        self._max_length = max(self.config.MIN_LENGTH_AFTER_DIVISION * 2, int(random_max_len))

        if not _from_division:
            for i in range(self.config.SEED_CELL_SEGMENTS):
                self._add_seed_cell_segments(i == 0)
            self._update_colors()


    def _add_seed_cell_segments(self, is_first: bool) -> None:
        """Adds a single new segment during the initial construction of the very first cell.

        This method creates a pymunk body and shape for one segment. It
        has two modes:
        1. If `is_first` is True, it places the segment at the cell's
           designated start position.
        2. If `is_first` is False, it places the segment adjacent to the
           previously added segment and creates the necessary physical joints
           (Pivot, and optionally Rotary Limit and Spring) to connect them.

        Args:
            is_first: A flag to indicate if this is the very first segment
                      of the cell.
        """

        segment = CellSegment(config = self.config, group_id=self.group_id)

        if is_first:
            segment.body.position = self.start_pos
        else:
            prev_body = self.bodies[-1]
            offset = Vec2d(self.joint_distance, 0).rotated(prev_body.angle)
            segment.body.position = prev_body.position + offset
            segment.body.angle = prev_body.angle
            noise_x = np.random.uniform(-0.1, 0.1)
            noise_y = np.random.uniform(-0.1, 0.1)
            segment.body.position += Vec2d(noise_x, noise_y)



        self.space.add(segment.body, segment.shape)
        self.bodies.append(segment.body)
        self.shapes.append(segment.shape)

        if not is_first:
            prev_body = self.bodies[-2]

            anchor_on_prev = (self.joint_distance / 2, 0)
            anchor_on_curr = (-self.joint_distance / 2, 0)
            pivot = pymunk.PivotJoint(
                prev_body, segment.body, anchor_on_prev, anchor_on_curr
            )
            pivot.max_force = self.config.PIVOT_JOINT_STIFFNESS # - pivots should have a hardcoded max force I think?
            self.space.add(pivot)

            if self.config.ROTARY_LIMIT_JOINT:
                assert self.config.STIFFNESS is not None
                assert self.config.MAX_BEND_ANGLE is not None
                limit = pymunk.RotaryLimitJoint(
                    prev_body, segment.body, -self.config.MAX_BEND_ANGLE, self.config.MAX_BEND_ANGLE
                )
                limit.max_force = self.config.STIFFNESS
                self.space.add(limit)
                self.limit_joints.append(limit)

            if self.config.DAMPED_ROTARY_SPRING:
                assert self.config.ROTARY_SPRING_STIFFNESS is not None
                assert self.config.ROTARY_SPRING_DAMPING is not None
                spring = pymunk.DampedRotarySpring(
                    prev_body, segment.body, 0, self.config.ROTARY_SPRING_STIFFNESS, self.config.ROTARY_SPRING_DAMPING
                )
                self.space.add(spring)
                self.spring_joints.append(spring)

            self.pivot_joints.append(pivot)

    def grow(self, dt):
        if not self.is_dividing and len(self.bodies) >= self._max_length:
            return

        if len(self.bodies) < 2:
            return

        self.growth_accumulator += (
                self.config.GROWTH_RATE * dt * np.random.uniform(0, 4)
        )

        last_pivot_joint = self.pivot_joints[-1]
        original_anchor_x = -self.joint_distance / 2
        last_pivot_joint.anchor_b = (
            original_anchor_x - self.growth_accumulator,
            0,
        )

        if self.growth_accumulator >= self.growth_threshold:
            pre_tail_body = self.bodies[-2]
            old_tail_body = self.bodies[-1]
            last_pivot_joint.anchor_b = (original_anchor_x, 0)
            stable_offset = Vec2d(self.joint_distance, 0).rotated(pre_tail_body.angle)
            old_tail_body.position = pre_tail_body.position + stable_offset
            old_tail_body.angle = pre_tail_body.angle

            moment = pymunk.moment_for_circle(self.config.SEGMENT_MASS, 0, self.config.SEGMENT_RADIUS)
            new_tail_body = pymunk.Body(self.config.SEGMENT_MASS, moment)
            new_tail_offset = Vec2d(self.joint_distance, 0).rotated(old_tail_body.angle)
            new_tail_body.position = old_tail_body.position + new_tail_offset
            new_tail_body.angle = old_tail_body.angle

            noise_x = np.random.uniform(-0.1, 0.1)
            noise_y = np.random.uniform(-0.1, 0.1)
            new_tail_body.position += Vec2d(noise_x, noise_y)
            new_tail_shape = pymunk.Circle(new_tail_body, self.config.SEGMENT_RADIUS)
            new_tail_shape.friction = 0.0
            new_tail_shape.filter = pymunk.ShapeFilter(group=self.group_id)
            self.space.add(new_tail_body, new_tail_shape)
            self.bodies.append(new_tail_body)
            self.shapes.append(new_tail_shape)

            anchor_on_prev = (self.joint_distance / 2, 0)
            anchor_on_curr = (-self.joint_distance / 2, 0)
            new_pivot = pymunk.PivotJoint(
                old_tail_body, new_tail_body, anchor_on_prev, anchor_on_curr
            )
            new_pivot.max_force = self.config.PIVOT_JOINT_STIFFNESS
            self.space.add(new_pivot)



            # --- REFACTORED: Append to specific joint lists ---
            self.pivot_joints.append(new_pivot)


            if self.config.ROTARY_LIMIT_JOINT:
                assert self.config.STIFFNESS is not None
                assert self.config.MAX_BEND_ANGLE is not None
                new_limit = pymunk.RotaryLimitJoint(
                    old_tail_body, new_tail_body, -self.config.MAX_BEND_ANGLE, self.config.MAX_BEND_ANGLE
                )
                new_limit.max_force = self.config.STIFFNESS
                self.space.add(new_limit)
                self.limit_joints.append(new_limit)

            if self.config.DAMPED_ROTARY_SPRING:
                assert self.config.ROTARY_SPRING_STIFFNESS is not None
                assert self.config.ROTARY_SPRING_DAMPING is not None
                new_spring = pymunk.DampedRotarySpring(
                    old_tail_body, new_tail_body, 0, self.config.ROTARY_SPRING_STIFFNESS,
                    self.config.ROTARY_SPRING_DAMPING
                )
                self.space.add(new_spring)
                # --- REFACTORED: Append to specific joint list ---
                self.spring_joints.append(new_spring)

            self.growth_accumulator = 0.0
            self._update_colors()

    def remove_tail_segment(self):
        """Safely removes the last segment of the cell."""
        if len(self.bodies) <= self.config.MIN_LENGTH_AFTER_DIVISION:
            return

        tail_body = self.bodies.pop()
        tail_shape = self.shapes.pop()

        # --- REFACTORED: Remove one joint from each relevant list ---
        if self.pivot_joints:
            self.space.remove(self.pivot_joints.pop())
        if self.limit_joints:
            self.space.remove(self.limit_joints.pop())
        if self.config.DAMPED_ROTARY_SPRING and self.spring_joints:
            self.space.remove(self.spring_joints.pop())

        self.space.remove(tail_body, tail_shape)
        self._update_colors()

    def _split_cell(self, next_group_id: int) -> Optional['Cell']:
        """Splits the current cell into two distinct cells if septum formation is complete."""
        progress = min(1.0, self.septum_progress)
        assert self.division_site is not None # To keep mypy happy
        for i in range(self.num_septum_segments):
            mother_idx = self.division_site - 1 - i
            daughter_idx = self.division_site + i
            if mother_idx >= 0 and daughter_idx < len(self.shapes):
                falloff = (self.num_septum_segments - i) / self.num_septum_segments
                shrinkage = (self.config.SEGMENT_RADIUS - self.min_septum_radius) * progress * falloff
                new_radius = self.config.SEGMENT_RADIUS - shrinkage
                self.shapes[mother_idx] = self._recreate_shape(self.shapes[mother_idx], new_radius)
                self.shapes[daughter_idx] = self._recreate_shape(self.shapes[daughter_idx], new_radius)
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
            space=self.space, config=self.config, start_pos=self.bodies[mother_final_len].position,
            group_id=next_group_id, _from_division=True
        )

        daughter_cell.bodies = self.bodies[mother_final_len:]
        daughter_cell.shapes = self.shapes[mother_final_len:]
        for shape in daughter_cell.shapes:
            shape.filter = pymunk.ShapeFilter(group=next_group_id)

        # --- REFACTORED: Partition and manage joint lists separately ---
        # The joint connecting mother and daughter is at index `mother_final_len - 1`
        connecting_joint_idx = mother_final_len - 1

        # 1. Assign the daughter's joints (all joints after the connection point)
        daughter_cell.pivot_joints = self.pivot_joints[mother_final_len:]
        if self.config.ROTARY_LIMIT_JOINT:
            daughter_cell.limit_joints = self.limit_joints[mother_final_len:]
        if self.config.DAMPED_ROTARY_SPRING:
            daughter_cell.spring_joints = self.spring_joints[mother_final_len:]

        # 2. Remove the connecting joints from the physics space
        self.space.remove(self.pivot_joints[connecting_joint_idx])
        if self.config.ROTARY_LIMIT_JOINT:
            self.space.remove(self.limit_joints[connecting_joint_idx])
        if self.config.DAMPED_ROTARY_SPRING:
            self.space.remove(self.spring_joints[connecting_joint_idx])

        # 3. Trim the mother's components
        self.bodies = self.bodies[:mother_final_len]
        self.shapes = self.shapes[:mother_final_len]
        self.pivot_joints = self.pivot_joints[:connecting_joint_idx]
        if self.config.ROTARY_LIMIT_JOINT:
            self.limit_joints = self.limit_joints[:connecting_joint_idx]
        if self.config.DAMPED_ROTARY_SPRING:
            self.spring_joints = self.spring_joints[:connecting_joint_idx]

        self._update_colors()
        daughter_cell._update_colors()

        return daughter_cell

    def apply_noise(self, dt: float):
        for body in self.bodies:
            force_x = np.random.uniform(-self.config.NOISE_STRENGTH, self.config.NOISE_STRENGTH)
            force_y = np.random.uniform(-self.config.NOISE_STRENGTH, self.config.NOISE_STRENGTH)
            body.force += Vec2d(force_x, force_y)
            torque = np.random.uniform(-self.config.NOISE_STRENGTH * 0.1, self.config.NOISE_STRENGTH * 0.1)
            body.torque += torque

    def divide(self, next_group_id: int, dt: float) -> Optional['Cell']:
        if not self.is_dividing:
            if len(self.bodies) < self._max_length:
                return None
            split_index = len(self.bodies) // 2
            if split_index < self.config.MIN_LENGTH_AFTER_DIVISION or \
                    (len(self.bodies) - split_index) < self.config.MIN_LENGTH_AFTER_DIVISION:
                return None
            self.is_dividing = True
            self.septum_progress = 0.0
            self.division_site = split_index
            self.length_at_division_start = len(self.bodies)
            return None
        if self.is_dividing:
            self.septum_progress += dt / self.septum_duration
            daughter_cell = self._split_cell(next_group_id)
            if daughter_cell:
                self.is_dividing = False
                self.septum_progress = 0.0
                self.division_site = None
                self.length_at_division_start = 0
            return daughter_cell
        return None

    def _update_colors(self):
        if not self.shapes: return
        r, g, b = self.base_color
        body_color = (r, g, b, 255)
        head_color = (min(255, int(r * 1.3)), min(255, int(g * 1.3)), min(255, int(b * 1.3)), 255)
        tail_color = (int(r * 0.7), int(g * 0.7), int(b * 0.7), 255)
        for shape in self.shapes: shape.color = body_color
        self.shapes[0].color = head_color
        self.shapes[-1].color = tail_color

    def _recreate_shape(self, shape_to_replace: pymunk.Circle, new_radius: float) -> pymunk.Circle:
        body = shape_to_replace.body
        friction, filter, color = shape_to_replace.friction, shape_to_replace.filter, shape_to_replace.color
        self.space.remove(shape_to_replace)
        new_shape = pymunk.Circle(body, new_radius)
        new_shape.friction, new_shape.filter, new_shape.color = friction, filter, color
        self.space.add(new_shape)
        return new_shape
