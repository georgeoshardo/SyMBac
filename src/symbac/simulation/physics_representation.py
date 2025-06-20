import pymunk
from pymunk import Vec2d
import numpy as np
from symbac.simulation.segments import CellSegment
from symbac.simulation.joints import CellJoint, CellRotaryLimitJoint, CellDampedRotarySpring
import pymunk

from symbac.simulation.config import CellConfig


class PhysicsRepresentation:
    """
    A class to hold all the cell's physics objects, such as segments, joints, and other physics-related attributes.
    """

    def __init__(self, space: pymunk.Space, config: CellConfig, group_id: int, start_pos: tuple[float, float]) -> None:
        """
        Initialize the PhysicsRepresentation with a Cell object.

        Parameters
        ----------


        """

        self.space = space
        self.group_id = group_id
        self.config = config
        self.start_pos = start_pos

        self.segments: list[CellSegment] = []

        self.pivot_joints: list[pymunk.PivotJoint] = []
        self.limit_joints: list[pymunk.RotaryLimitJoint] = []
        self.spring_joints: list[pymunk.DampedRotarySpring] = []

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

        config = self.config

        if is_first:
            segment = CellSegment(config=config, group_id=self.cell.group_id, position=self.cell.start_pos, space=self.space)
        else:
            prev_segment = self.segments[-1]
            offset = Vec2d(self.config.JOINT_DISTANCE, 0).rotated(prev_segment.angle)

            noise_x = np.random.uniform(-0.1, 0.1)
            noise_y = np.random.uniform(-0.1, 0.1)
            segment_position = prev_segment.position[0] + offset[0] + noise_x, prev_segment.position[1] + offset[1] + noise_y
            segment_angle = prev_segment.angle
            segment = CellSegment(config=self.config, group_id=self.cell.group_id, position=segment_position,
                                  angle=segment_angle, space=self.space)

        self.space.add(segment.body, segment.shape)
        self.segments.append(segment)

        if not is_first:
            prev_segment = self.segments[-2]
            joint = CellJoint(prev_segment, segment, config)
            self.space.add(joint)
            self.pivot_joints.append(joint)

            if config.ROTARY_LIMIT_JOINT:
                limit = CellRotaryLimitJoint(prev_segment, segment, config)
                self.space.add(limit)
                self.limit_joints.append(limit)

            if config.DAMPED_ROTARY_SPRING:
                spring = CellDampedRotarySpring(prev_segment, segment, config)
                self.space.add(spring)
                self.spring_joints.append(spring)

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
            group_id=self.cell.group_id,
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