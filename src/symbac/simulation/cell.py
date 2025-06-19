import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np
from typing import Optional
from symbac.misc import generate_color
from symbac.simulation.config import CellConfig
from symbac.simulation.segments import CellSegment
from symbac.simulation.joints import CellJoint, CellRotaryLimitJoint, CellDampedRotarySpring


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

        self.segments: list[CellSegment] = []

        self.pivot_joints: list[pymunk.PivotJoint] = []
        self.limit_joints: list[pymunk.RotaryLimitJoint] = []
        self.spring_joints: list[pymunk.DampedRotarySpring] = []

        self.growth_accumulator_head = 0.0
        self.growth_accumulator_tail = 0.0
        self.is_dividing = False
        self.septum_progress = 0.0
        self.septum_duration = 1.5
        self.division_site: int | None = None
        self.num_septum_segments = self.config.GRANULARITY - 1
        self.min_septum_radius = self.config.SEGMENT_RADIUS * 0.25
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

        This method creates a CellSegment. It has two modes:
        1. If `is_first` is True, it places the segment at the cell's
           designated start position.
        2. If `is_first` is False, it places the segment adjacent to the
           previously added segment and creates the necessary physical joints
           (Pivot, and optionally Rotary Limit and Spring) to connect them.

        Args:
            is_first: A flag to indicate if this is the very first segment
                      of the cell.
        """

        if is_first:
            segment = CellSegment(config=self.config, group_id=self.group_id, position=self.start_pos, space=self.space)
        else:
            prev_segment = self.segments[-1]
            offset = Vec2d(self.config.JOINT_DISTANCE, 0).rotated(prev_segment.angle)

            noise_x = np.random.uniform(-0.1, 0.1)
            noise_y = np.random.uniform(-0.1, 0.1)
            segment_position = prev_segment.position[0] + offset[0] + noise_x, prev_segment.position[1] + offset[1] + noise_y
            segment_angle = prev_segment.angle
            segment = CellSegment(config=self.config, group_id=self.group_id, position=segment_position,
                                  angle=segment_angle, space=self.space)

        self.space.add(segment.body, segment.shape)
        self.segments.append(segment)

        if not is_first:
            prev_segment = self.segments[-2]
            joint = CellJoint(prev_segment, segment, self.config)
            self.space.add(joint)
            self.pivot_joints.append(joint)

            if self.config.ROTARY_LIMIT_JOINT:
                limit = CellRotaryLimitJoint(prev_segment, segment, self.config)
                self.space.add(limit)
                self.limit_joints.append(limit)

            if self.config.DAMPED_ROTARY_SPRING:
                spring = CellDampedRotarySpring(prev_segment, segment, self.config)
                self.space.add(spring)
                self.spring_joints.append(spring)

    def grow(self, dt):
        if not self.is_dividing and len(self.segments) >= self._max_length:
            return

        if len(self.segments) < 2:
            return

        added_length = self.config.GROWTH_RATE * dt * np.random.uniform(0, 4)

        half_growth = added_length / 2

        self.growth_accumulator_head += (
            half_growth
        ) # Head increases in length by half the total added length

        self.growth_accumulator_tail += (
            half_growth
        )  # Tail increases in length by half the total added length

        # Adjust and stretch the head joint anchor
        first_pivot_joint = self.pivot_joints[0]
        first_pivot_joint.anchor_a = (
            self.config.JOINT_DISTANCE / 2 + self.growth_accumulator_head,
            0
        )

        # Adjust and stretch the tail joint anchor
        last_pivot_joint = self.pivot_joints[-1]
        last_pivot_joint.anchor_b = (
            -self.config.JOINT_DISTANCE / 2 - self.growth_accumulator_tail,
            0,
        )

        if self.growth_accumulator_head >= self.config.GROWTH_THRESHOLD:
            first_pivot_joint.anchor_a = (self.config.JOINT_DISTANCE / 2, 0)
            self._add_head_segment()
            self.growth_accumulator_head = 0.0
            self._update_colors()

        if self.growth_accumulator_tail >= self.config.GROWTH_THRESHOLD:
            last_pivot_joint.anchor_b = (-self.config.JOINT_DISTANCE / 2, 0)
            self._add_tail_segment()
            self.growth_accumulator_tail = 0.0
            self._update_colors()

    def _add_head_segment(self):
        """Adds a new segment to the head of the cell."""
        # This is the mirror logic of _add_segment_to_tail
        old_head_segment = self.segments[0]
        post_head_segment = self.segments[1]

        # Stabilize the old head segment
        stable_offset = Vec2d(-self.config.JOINT_DISTANCE, 0).rotated(post_head_segment.angle)
        old_head_segment.position = post_head_segment.position + stable_offset
        old_head_segment.angle = post_head_segment.angle

        # Calculate position for the new head segment
        new_head_offset = Vec2d(-self.config.JOINT_DISTANCE, 0).rotated(old_head_segment.angle)
        new_position = old_head_segment.position + new_head_offset
        new_position = new_position[0], new_position[1] # Convert Vec2d to tuple

        new_head_segment = CellSegment(
            config=self.config,
            group_id=self.group_id,
            position=new_position,
            angle=old_head_segment.angle,
            space=self.space
        )

        self.space.add(new_head_segment.body, new_head_segment.shape)
        self.segments.insert(0, new_head_segment)

        # Add joints connecting the new head to the old head
        new_pivot = CellJoint(new_head_segment, old_head_segment, self.config)
        self.space.add(new_pivot)
        self.pivot_joints.insert(0, new_pivot)

        if self.config.ROTARY_LIMIT_JOINT:
            new_limit = CellRotaryLimitJoint(new_head_segment, old_head_segment, self.config)
            self.space.add(new_limit)
            self.limit_joints.insert(0, new_limit)

        if self.config.DAMPED_ROTARY_SPRING:
            new_spring = CellDampedRotarySpring(new_head_segment, old_head_segment, self.config)
            self.space.add(new_spring)
            self.spring_joints.insert(0, new_spring)

    def _add_tail_segment(self):
        pre_tail_segment = self.segments[-2]
        old_tail_segment = self.segments[-1]

        stable_offset = Vec2d(self.config.JOINT_DISTANCE, 0).rotated(pre_tail_segment.angle)
        old_tail_segment.position = pre_tail_segment.position + stable_offset
        old_tail_segment.angle = pre_tail_segment.angle

        new_tail_offset = Vec2d(self.config.JOINT_DISTANCE, 0).rotated(old_tail_segment.angle)
        new_position = old_tail_segment.position + new_tail_offset

        noise_x = np.random.uniform(-0.1, 0.1)
        noise_y = np.random.uniform(-0.1, 0.1)
        new_position += Vec2d(noise_x, noise_y)
        new_position = new_position[0], new_position[1]  # Convert Vec2d to tuple for mypy

        new_tail_segment = CellSegment(
            config=self.config,
            group_id=self.group_id,
            position=new_position,
            angle=old_tail_segment.angle,
            space=self.space,
        )

        self.space.add(new_tail_segment.body, new_tail_segment.shape)
        self.segments.append(new_tail_segment)

        # Add joints connecting the old tail to the new tail
        new_pivot = CellJoint(old_tail_segment, new_tail_segment, self.config)
        self.space.add(new_pivot)
        self.pivot_joints.append(new_pivot)

        if self.config.ROTARY_LIMIT_JOINT:
            new_limit = CellRotaryLimitJoint(old_tail_segment, new_tail_segment, self.config)
            self.space.add(new_limit)
            self.limit_joints.append(new_limit)

        if self.config.DAMPED_ROTARY_SPRING:
            new_spring = CellDampedRotarySpring(old_tail_segment, new_tail_segment, self.config)
            self.space.add(new_spring)
            self.spring_joints.append(new_spring)

    def remove_tail_segment(self):
        """Safely removes the last segment of the cell."""
        if len(self.segments) <= self.config.MIN_LENGTH_AFTER_DIVISION:
            return

        tail_segment = self.segments.pop()

        if self.pivot_joints:
            self.space.remove(self.pivot_joints.pop())
        if self.limit_joints:
            self.space.remove(self.limit_joints.pop())
        if self.config.DAMPED_ROTARY_SPRING and self.spring_joints:
            self.space.remove(self.spring_joints.pop())

        self.space.remove(tail_segment.body, tail_segment.shape)
        self._update_colors()

    def _split_cell(self, next_group_id: int) -> Optional['Cell']:
        """Splits the current cell into two distinct cells if septum formation is complete."""
        progress = min(1.0, self.septum_progress)
        assert self.division_site is not None  # To keep mypy happy
        for i in range(self.num_septum_segments):
            mother_idx = self.division_site - 1 - i
            daughter_idx = self.division_site + i
            if mother_idx >= 0 and daughter_idx < len(self.segments):
                falloff = (self.num_septum_segments - i) / self.num_septum_segments
                shrinkage = (self.config.SEGMENT_RADIUS - self.min_septum_radius) * progress * falloff
                new_radius = self.config.SEGMENT_RADIUS - shrinkage

                # Recreate shape and update the segment's shape reference
                self.segments[mother_idx].radius = new_radius
                self.segments[daughter_idx].radius = new_radius

        if progress < 1.0:
            return None

        # Restore original radius after split
        for i in range(self.num_septum_segments):
            mother_idx = self.division_site - 1 - i
            daughter_idx = self.division_site + i
            if mother_idx >= 0 and daughter_idx < len(self.segments):
                self.segments[mother_idx].radius = self.config.SEGMENT_RADIUS
                self.segments[daughter_idx].radius = self.config.SEGMENT_RADIUS
        current_length = len(self.segments)
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
            space=self.space, config=self.config, start_pos=self.segments[mother_final_len].position,
            group_id=next_group_id, _from_division=True
        )

        daughter_cell.segments = self.segments[mother_final_len:]
        for segment in daughter_cell.segments:
            segment.shape.filter = pymunk.ShapeFilter(group=next_group_id)

        connecting_joint_idx = mother_final_len - 1

        # 1. Assign the daughter's joints
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
        self.segments = self.segments[:mother_final_len]
        self.pivot_joints = self.pivot_joints[:connecting_joint_idx]
        if self.config.ROTARY_LIMIT_JOINT:
            self.limit_joints = self.limit_joints[:connecting_joint_idx]
        if self.config.DAMPED_ROTARY_SPRING:
            self.spring_joints = self.spring_joints[:connecting_joint_idx]

        self._update_colors()
        daughter_cell._update_colors()

        return daughter_cell

    def apply_noise(self, dt: float):
        for segment in self.segments:
            force_x = np.random.uniform(-self.config.NOISE_STRENGTH, self.config.NOISE_STRENGTH)
            force_y = np.random.uniform(-self.config.NOISE_STRENGTH, self.config.NOISE_STRENGTH)
            segment.body.force += Vec2d(force_x, force_y)
            torque = np.random.uniform(-self.config.NOISE_STRENGTH * 0.1, self.config.NOISE_STRENGTH * 0.1)
            segment.body.torque += torque

    def divide(self, next_group_id: int, dt: float) -> Optional['Cell']:
        if not self.is_dividing:
            if len(self.segments) < self._max_length:
                return None
            split_index = len(self.segments) // 2
            if split_index < self.config.MIN_LENGTH_AFTER_DIVISION or \
                    (len(self.segments) - split_index) < self.config.MIN_LENGTH_AFTER_DIVISION:
                return None
            self.is_dividing = True
            self.septum_progress = 0.0
            self.division_site = split_index
            self.length_at_division_start = len(self.segments)
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
        if not self.segments: return
        r, g, b = self.base_color
        body_color = (r, g, b, 255)
        head_color = (min(255, int(r * 1.3)), min(255, int(g * 1.3)), min(255, int(b * 1.3)), 255)
        tail_color = (int(r * 0.7), int(g * 0.7), int(b * 0.7), 255)
        for segment in self.segments:
            segment.shape.color = body_color
        self.segments[0].shape.color = head_color
        self.segments[-1].shape.color = tail_color
