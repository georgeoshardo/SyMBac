import pymunk
from pymunk import Vec2d
import numpy as np
from symbac.simulation.segments import CellSegment
from symbac.simulation.joints import CellJoint, CellRotaryLimitJoint, CellDampedRotarySpring
import pymunk
from typing import cast, Optional
from symbac.simulation.config import CellConfig


class PhysicsRepresentation:
    _daughter_septum_segments: Optional[list[CellSegment]] = None
    _mother_septum_segments: Optional[list[CellSegment]] = None
    """
    A class to hold all the cell's physics objects, such as segments, joints, and other physics-related attributes.
    Does not deal with division or growth directly, but provides methods to manipulate the segments and joints.
    """

    def __init__(self, space: pymunk.Space, config: CellConfig, group_id: int, start_pos: Vec2d, _from_division: bool = False) -> None:
        """
        Initialize the physics_representation with a Cell object.

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

        self.mother_septum_segments = None
        self.daughter_septum_segments = None

        self.growth_accumulator_head = 0
        self.growth_accumulator_tail = 0

        if not _from_division:
            for i in range(self.config.SEED_CELL_SEGMENTS):
                self.add_seed_cell_segments(i == 0) # True for the first segment, False for the rest


    def add_seed_cell_segments(self, is_first: bool) -> None:
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
            segment = CellSegment(config=self.config, group_id=self.group_id, position=self.start_pos, space=self.space, angle=self.config.START_ANGLE)
        else:
            prev_segment = self.segments[-1]
            offset = Vec2d(self.config.JOINT_DISTANCE, 0).rotated(prev_segment.angle)

            noise_x = np.random.uniform(-0.1, 0.1)
            noise_y = np.random.uniform(-0.1, 0.1)
            noise = Vec2d(noise_x, noise_y)
            segment_position = prev_segment.position + cast(tuple[float,float], offset) + cast(tuple[float,float], noise)
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

    def add_head_segment(self):
        """
        Inserts a new segment between the head (segment 0) and the segment next to it (segment 1).
        This method ensures the head segment does not "jump" but instead glides away smoothly.
        """
        # The head segment that should remain in place.
        final_head_segment = self.segments[0]
        # The segment that was previously connected to the head.
        post_head_segment = self.segments[1]

        # Calculate the position for the new segment. It will be placed one joint distance
        # away from the post_head_segment, in the direction of the final_head_segment.
        direction = (final_head_segment.position - cast(tuple[float,float], post_head_segment.position)).normalized() # Insanity to keep the type checker happy
        new_segment_pos = post_head_segment.position + cast(tuple[float,float], direction * self.config.JOINT_DISTANCE)
        new_segment_angle = post_head_segment.angle

        # Create the new segment to be inserted.
        new_segment = CellSegment(
            config=self.config, group_id=self.group_id, position=new_segment_pos,
            angle=new_segment_angle, space=self.space
        )
        self.space.add(new_segment.body, new_segment.shape)

        # Insert the new segment into the list at the correct position.
        self.segments.insert(1, new_segment)

        # --- Rewire the joints ---
        # 1. Remove the old, stretched joint between the original head and post-head segments.
        self.space.remove(self.pivot_joints.pop(0))
        if self.config.ROTARY_LIMIT_JOINT:
            self.space.remove(self.limit_joints.pop(0))
        if self.config.DAMPED_ROTARY_SPRING:
            self.space.remove(self.spring_joints.pop(0))

        # 2. Create a new joint between the final head and the new segment.
        joint1 = CellJoint(final_head_segment, new_segment, self.config)
        self.space.add(joint1)
        self.pivot_joints.insert(0, joint1)
        if self.config.ROTARY_LIMIT_JOINT:
            limit1 = CellRotaryLimitJoint(final_head_segment, new_segment, self.config)
            self.space.add(limit1)
            self.limit_joints.insert(0, limit1)
        if self.config.DAMPED_ROTARY_SPRING:
            spring1 = CellDampedRotarySpring(final_head_segment, new_segment, self.config)
            self.space.add(spring1)
            self.spring_joints.insert(0, spring1)

        # 3. Create a new joint between the new segment and the post-head segment.
        joint2 = CellJoint(new_segment, post_head_segment, self.config)
        self.space.add(joint2)
        self.pivot_joints.insert(1, joint2)
        if self.config.ROTARY_LIMIT_JOINT:
            limit2 = CellRotaryLimitJoint(new_segment, post_head_segment, self.config)
            self.space.add(limit2)
            self.limit_joints.insert(1, limit2)
        if self.config.DAMPED_ROTARY_SPRING:
            spring2 = CellDampedRotarySpring(new_segment, post_head_segment, self.config)
            self.space.add(spring2)
            self.spring_joints.insert(1, spring2)

    def add_tail_segment(self):
        """
        Inserts a new segment between the tail (last segment) and the one before it.
        This method ensures the tail segment does not "jump" but instead glides away smoothly.
        """
        # The tail segment that should remain in place.
        final_tail_segment = self.segments[-1]
        # The segment that was previously connected to the tail.
        pre_tail_segment = self.segments[-2]

        # Calculate the position for the new segment. It will be placed one joint distance
        # away from the pre_tail_segment, in the direction of the final_tail_segment.
        direction = (final_tail_segment.position - cast(tuple[float,float],pre_tail_segment.position)).normalized()
        new_segment_pos = Vec2d(*pre_tail_segment.position) + cast(tuple[float,float],direction * self.config.JOINT_DISTANCE)
        new_segment_angle = pre_tail_segment.angle

        # Create the new segment to be inserted.
        new_segment = CellSegment(
            config=self.config, group_id=self.group_id, position=new_segment_pos,
            angle=new_segment_angle, space=self.space
        )
        self.space.add(new_segment.body, new_segment.shape)

        # Insert the new segment into the list before the final tail segment.
        self.segments.insert(-1, new_segment)

        # --- Rewire the joints ---
        # 1. Remove the old, stretched joint between the pre-tail and final-tail segments.
        self.space.remove(self.pivot_joints.pop())
        if self.config.ROTARY_LIMIT_JOINT:
            self.space.remove(self.limit_joints.pop())
        if self.config.DAMPED_ROTARY_SPRING:
            self.space.remove(self.spring_joints.pop())

        # 2. Create a new joint between the pre-tail segment and the new segment.
        joint1 = CellJoint(pre_tail_segment, new_segment, self.config)
        self.space.add(joint1)
        self.pivot_joints.append(joint1)  # Append, since we just popped the last one.
        if self.config.ROTARY_LIMIT_JOINT:
            limit1 = CellRotaryLimitJoint(pre_tail_segment, new_segment, self.config)
            self.space.add(limit1)
            self.limit_joints.append(limit1)
        if self.config.DAMPED_ROTARY_SPRING:
            spring1 = CellDampedRotarySpring(pre_tail_segment, new_segment, self.config)
            self.space.add(spring1)
            self.spring_joints.append(spring1)

        # 3. Create a new joint between the new segment and the final tail segment.
        joint2 = CellJoint(new_segment, final_tail_segment, self.config)
        self.space.add(joint2)
        self.pivot_joints.append(joint2)  # This now becomes the new last joint.
        if self.config.ROTARY_LIMIT_JOINT:
            limit2 = CellRotaryLimitJoint(new_segment, final_tail_segment, self.config)
            self.space.add(limit2)
            self.limit_joints.append(limit2)
        if self.config.DAMPED_ROTARY_SPRING:
            spring2 = CellDampedRotarySpring(new_segment, final_tail_segment, self.config)
            self.space.add(spring2)
            self.spring_joints.append(spring2)

    def remove_tail_segment(self) -> Optional[CellSegment]:
        """Safely removes the last segment of the cell."""
        if len(self.segments) <= self.config.MIN_LENGTH_AFTER_DIVISION:
            return None
        assert len(self.limit_joints) == len(self.pivot_joints)
        assert len(self.limit_joints) == len(self.segments) - 1
        tail_segment = self.segments.pop()

        if self.pivot_joints:
            self.space.remove(self.pivot_joints.pop())
        if self.limit_joints:
            self.space.remove(self.limit_joints.pop())
        if self.config.DAMPED_ROTARY_SPRING and self.spring_joints:
            self.space.remove(self.spring_joints.pop())

        self.space.remove(tail_segment.body, tail_segment.shape)
        return tail_segment

    def remove_head_segment(self) -> Optional[CellSegment]:
        """Safely removes the first segment of the cell."""
        if len(self.segments) <= self.config.MIN_LENGTH_AFTER_DIVISION:
            return None

        head_segment = self.segments.pop(0)

        if self.pivot_joints:
            self.space.remove(self.pivot_joints.pop(0))
        if self.config.ROTARY_LIMIT_JOINT and self.limit_joints:
            self.space.remove(self.limit_joints.pop(0))
        if self.config.DAMPED_ROTARY_SPRING and self.spring_joints:
            self.space.remove(self.spring_joints.pop(0))

        self.space.remove(head_segment.body, head_segment.shape)
        return head_segment

    def apply_noise(self, dt: float):
        for segment in self.segments:
            force_x = np.random.uniform(-self.config.NOISE_STRENGTH, self.config.NOISE_STRENGTH)
            force_y = np.random.uniform(-self.config.NOISE_STRENGTH, self.config.NOISE_STRENGTH)
            segment.body.force += Vec2d(force_x, force_y)
            torque = np.random.uniform(-self.config.NOISE_STRENGTH * 0.1, self.config.NOISE_STRENGTH * 0.1)
            segment.body.torque += torque
            

    def check_joint_integrity(self, failure_threshold: float = 0.25) -> None:
        """
        Checks if pivot joints are failing to maintain segment spacing,
        especially under compression. It prints a warning if the distance
        between two segments is smaller than expected by a given threshold.

        Args:
            failure_threshold: The relative deviation from the expected
                               distance that triggers a warning (e.g., 0.25 for 25%).
        """
        if len(self.segments) < 2:
            return

        num_joints = len(self.pivot_joints)
        for i, joint in enumerate(self.pivot_joints):
            segment_a = self.segments[i]
            segment_b = self.segments[i + 1]

            actual_distance = (segment_b.position - cast(tuple[float,float],segment_a.position)).length

            expected_distance = self.config.JOINT_DISTANCE
            # The first joint is stretched by head growth
            if i == 0:
                expected_distance += self.growth_accumulator_head
            # The last joint is stretched by tail growth
            if i == num_joints - 1:
                expected_distance += self.growth_accumulator_tail

            # Check for compression (actual distance is less than expected)
            if actual_distance < expected_distance:
                deviation = expected_distance - actual_distance
                if deviation / expected_distance > failure_threshold:
                    print(
                        f"WARNING: Joint {i} in cell {self.group_id} is under high compression! "
                        f"Expected: {expected_distance:.2f}, "
                        f"Actual: {actual_distance:.2f}, "
                        f"Deviation: {deviation:.2f}"
                    )

    def get_compression_ratio(self) -> float:
        """
        Calculates the total expected length of the cell and compares it to the
        actual continuous length, printing a warning if the cell is compressed
        beyond a given threshold.

        Args:
            compression_threshold: The relative deviation from the expected
                                   length that triggers a warning (e.g., 0.10 for 10%).
        """
        if len(self.segments) < 2:
            raise Exception("Cell is fewer than 2 segments!")

        # Calculate the total expected length between the centers of the first and last segments
        num_joints = len(self.pivot_joints)
        expected_internal_length = (num_joints * self.config.JOINT_DISTANCE) + \
                                   self.growth_accumulator_head + \
                                   self.growth_accumulator_tail

        # Add radii for tip-to-tip length
        expected_total_length = expected_internal_length + self.segments[0].radius + self.segments[-1].radius

        # Get the actual tip-to-tip length
        actual_total_length = self.get_continuous_length()

        return actual_total_length / expected_total_length

                
    def get_continuous_length(self) -> float:
        """Calculates the continuous length of the cell from tip to tip."""
        if not self.segments:
            return 0.0

        if len(self.segments) == 1:
            return self.segments[0].radius * 2

        total_length = 0.0
        for i in range(1, len(self.segments)): # TODO use pymunk batching to get this data
            segment_a = self.segments[i - 1]
            segment_b = self.segments[i]
            distance = segment_b.position - cast(tuple[float,float],segment_a.position)
            total_length += distance.length

        # Add the radius of the first and last segments to get the tip-to-tip length
        total_length += self.segments[0].radius + self.segments[-1].radius
        return total_length
