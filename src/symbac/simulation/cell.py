import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np
from typing import Optional
import sys
sys.path.insert(0, '..')
from misc import generate_color
from config import CellConfig
from segments import CellSegment
from joints import CellJoint, CellRotaryLimitJoint, CellDampedRotarySpring
import colorsys

# Note that length units here are the number of spheres in the cell, TODO: implement the continuous length measurement for rendering.
class Cell:
    _daughter_septum_segments: Optional[list[CellSegment]] = None
    _mother_septum_segments: Optional[list[CellSegment]] = None
    def __init__(
            self,
            space: pymunk.Space,
            config: CellConfig,
            start_pos: tuple[float, float],
            group_id: int = 0,
            _from_division: bool = False,
            base_color: Optional[tuple[int, int, int]] = None
    ) -> None:
        self.space = space
        self.config = config
        self.start_pos = start_pos
        self.group_id = group_id
        if not base_color:
            self.base_color = generate_color(group_id)
        else:
            self.base_color = base_color

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
        self.num_septum_segments = self.config.GRANULARITY
        self.min_septum_radius = self.config.SEGMENT_RADIUS * 0.1
        self.length_at_division_start = 0
        self.division_bias = 0 #self.config.GRANULARITY # not needed when bidirectional growth I think

        variation = self.config.BASE_MAX_LENGTH * self.config.MAX_LENGTH_VARIATION
        random_max_len = np.random.uniform(
            self.config.BASE_MAX_LENGTH - variation, self.config.BASE_MAX_LENGTH + variation
        )

        self._max_length = max(self.config.MIN_LENGTH_AFTER_DIVISION * 2, int(random_max_len))

        self.adjusted_growth_rate = self.config.GROWTH_RATE
        self.check_total_compression()

        self.num_divisions = 0

        if not _from_division:
            for i in range(self.config.SEED_CELL_SEGMENTS):
                self._add_seed_cell_segments(i == 0)
            self._update_colors()

        self._mother_septum_segments = None
        self._daughter_septum_segments = None


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

        added_length = self.adjusted_growth_rate * dt * np.random.uniform(0, 4)

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
            #print("added head segment, current length:", len(self.segments))

        if self.growth_accumulator_tail >= self.config.GROWTH_THRESHOLD:
            last_pivot_joint.anchor_b = (-self.config.JOINT_DISTANCE / 2, 0)
            self._add_tail_segment()
            self.growth_accumulator_tail = 0.0
            #print("added tail segment, current length:", len(self.segments))

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
        self._update_colors()
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
        self._update_colors()
        return head_segment

    def _split_cell(self, next_group_id: int) -> Optional['Cell']:
        """Splits the current cell into two distinct cells if septum formation is complete."""
        progress = min(1.0, self.septum_progress)
        self.division_site = len(self.segments) // 2 # Dynamically determine the division site but TODO it has some instability when the segments are being added during division

        if self._mother_septum_segments is None and self._daughter_septum_segments is None:
            self._mother_septum_segments = []
            self._daughter_septum_segments = []
            for i in range(self.num_septum_segments):
                mother_idx = self.division_site - 1 - i
                daughter_idx = self.division_site + i
                if mother_idx >= 0 and daughter_idx < len(self.segments):
                    # Recreate shape and update the segment's shape reference
                    self._mother_septum_segments.append(self.segments[mother_idx])
                    self._daughter_septum_segments.append(self.segments[daughter_idx])

        if self._mother_septum_segments is not None and self._daughter_septum_segments is not None:
            for i in range(self.num_septum_segments):
                falloff = (self.num_septum_segments - i) / self.num_septum_segments
                shrinkage = (self.config.SEGMENT_RADIUS - self.min_septum_radius) * progress * falloff
                new_radius = self.config.SEGMENT_RADIUS - shrinkage

                # Recreate shape and update the segment's shape reference
                self._mother_septum_segments[i].radius = new_radius
                self._daughter_septum_segments[i].radius = new_radius

        if progress < 1.0:
            return None

        # Restore original radius after split
        assert self._mother_septum_segments is not None
        assert self._daughter_septum_segments is not None
        for i in range(self.num_septum_segments):
            self._mother_septum_segments[i].radius = self.config.SEGMENT_RADIUS
            self._daughter_septum_segments[i].radius = self.config.SEGMENT_RADIUS

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

        self._mother_septum_segments = None
        self._daughter_septum_segments = None
        # --- START of MODIFIED SECTION for color inheritance ---

        # 1. Get the mother's color and normalize it to the 0-1 range for colorsys
        r, g, b = self.base_color
        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0

        # 2. Convert RGB to HSV
        h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
        # 3. Mutate the Hue to change the color while preserving lineage
        #    A small hue shift changes the color along the color wheel (e.g., red -> orange)
        hue_shift = np.random.uniform(-1, 1) / (np.sqrt(next_group_id) * 2)  # Shift hue with biased rw
        #s_shift = np.random.uniform(-0.2, 0.2) / (np.sqrt(next_group_id) / 1.8)  # Slight saturation shift
        #v_shift = np.random.uniform(-0.2, 0.2) / (np.sqrt(next_group_id) / 1.8) # Slight brightness shift
        new_h = (h + hue_shift) % 1.0  # Use modulo to wrap around the color wheel
        #    This prevents colors from becoming grayish or dark.
        #    We'll clamp them to a minimum vibrancy level.
        new_s = s
        new_v = v

        # 5. Convert the new HSV color back to RGB
        new_r, new_g, new_b = colorsys.hsv_to_rgb(new_h, new_s, new_v)

        # 6. Scale back to 0-255 and create the final tuple
        daughter_color = (int(new_r * 255), int(new_g * 255), int(new_b * 255))


        daughter_cell = Cell(
            space=self.space, config=self.config, start_pos=self.segments[mother_final_len].position,
            group_id=next_group_id, _from_division=True, base_color=daughter_color
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

        daughter_cell.growth_accumulator_tail = self.growth_accumulator_tail

        self.growth_accumulator_tail = 0.0
        self.num_divisions += 1

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

    def _update_colors(self) -> None:
        if not self.segments: return
        a = 255
        r, g, b = self.base_color

        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0

        # 2. Convert RGB to HSV
        h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
        new_s = max(s / np.sqrt(self.num_divisions+1), 0.3)  # Ensure saturation is not too low
        new_v = max(v / np.sqrt(self.num_divisions+1), 0.3) # Ensure brightness is not too low

        # 5. Convert the new HSV color back to RGB
        r, g, b = colorsys.hsv_to_rgb(h, new_s, new_v)
        r, g, b = (int(r * 255), int(g * 255), int(b * 255))

        body_color = (r,g,b,a)
        head_color = (min(255, int(r * 1.3)), min(255, int(g * 1.3)), min(255, int(b * 1.3)), a)
        tail_color = (int(r * 0.7), int(g * 0.7), int(b * 0.7), a)
        for segment in self.segments:
            segment.shape.color = body_color
        self.segments[0].shape.color = head_color
        self.segments[-1].shape.color = tail_color

    def get_continuous_length(self) -> float:
        """Calculates the continuous length of the cell from tip to tip."""
        if not self.segments:
            return 0.0

        if len(self.segments) == 1:
            return self.segments[0].radius * 2

        total_length = 0.0
        for i in range(1, len(self.segments)):
            segment_a = self.segments[i - 1]
            segment_b = self.segments[i]
            distance = Vec2d(*segment_b.position) - Vec2d(*segment_a.position)
            total_length += distance.length

        # Add the radius of the first and last segments to get the tip-to-tip length
        total_length += self.segments[0].radius + self.segments[-1].radius
        return total_length

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

            actual_distance = (Vec2d(*segment_b.position) - Vec2d(*segment_a.position)).length

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

    def check_total_compression(self, compression_threshold: float = 0.999) -> None:
        """
        Calculates the total expected length of the cell and compares it to the
        actual continuous length, printing a warning if the cell is compressed
        beyond a given threshold.

        Args:
            compression_threshold: The relative deviation from the expected
                                   length that triggers a warning (e.g., 0.10 for 10%).
        """
        if len(self.segments) < 2:
            return

        # Calculate the total expected length between the centers of the first and last segments
        num_joints = len(self.pivot_joints)
        expected_internal_length = (num_joints * self.config.JOINT_DISTANCE) + \
                                   self.growth_accumulator_head + \
                                   self.growth_accumulator_tail

        # Add radii for tip-to-tip length
        expected_total_length = expected_internal_length + self.segments[0].radius + self.segments[-1].radius

        # Get the actual tip-to-tip length
        actual_total_length = self.get_continuous_length()

        self.adjusted_growth_rate = min(self.config.GROWTH_RATE *  (actual_total_length / expected_total_length)**4, self.config.GROWTH_RATE)

        # Check for overall cell compression
        if actual_total_length < expected_total_length:
            deviation = expected_total_length - actual_total_length
            if (deviation / expected_total_length) > compression_threshold:
                print(
                    f"WARNING: Cell {self.group_id} is under high compression! "
                    f"Expected Length: {expected_total_length:.2f}, "
                    f"Actual Length: {actual_total_length:.2f}, "
                    f"Deviation: {deviation:.2f}",
                    f"Fractional Growth Rate: {self.adjusted_growth_rate / self.config.GROWTH_RATE:.2f}"
                )