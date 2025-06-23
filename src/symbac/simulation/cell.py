import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np
from typing import Optional, cast
from symbac.misc import generate_color
from symbac.simulation.config import CellConfig
from symbac.simulation.division_manager import DivisionManager
from symbac.simulation.physics_representation import PhysicsRepresentation
from symbac.simulation.segments import CellSegment
import colorsys

# Note that length units here are the number of spheres in the cell, TODO: implement the continuous length measurement for rendering.
class Cell:
    _daughter_septum_segments: Optional[list[CellSegment]] = None
    _mother_septum_segments: Optional[list[CellSegment]] = None


    def __init__(
            self,
            space: pymunk.Space,
            config: CellConfig,
            start_pos: Vec2d,
            group_id: int = 0,
            _from_division: bool = False,
            base_color: Optional[tuple[int, int, int]] = None
    ) -> None:

        if isinstance(start_pos, tuple):
            start_pos = Vec2d(*start_pos)

        self.space = space
        self.config = config
        self.start_pos = start_pos
        self._group_id = group_id
        if not base_color:
            self.base_color = generate_color(group_id)
        else:
            self.base_color = base_color

        self.PhysicsRepresentation = PhysicsRepresentation(
            space=self.space,
            config=self.config,
            group_id=self._group_id,
            start_pos=self.start_pos,
            _from_division=_from_division,
        ) # Use dependency injection pattern for the physics representation, can't think of a better way

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

        self.num_divisions = 0

        self._update_colors()

        self._mother_septum_segments = None
        self._daughter_septum_segments = None

    @property
    def group_id(self) -> int:
        """Returns the group ID of the cell, but it's immutable now."""
        return self._group_id

    @group_id.setter
    def group_id(self, value: int) -> None:
        """
        Tell the user they shouldn't be trying to meddle with the group_id
        """
        raise AttributeError("Cell lineage ID (group_id) is immutable and cannot be changed after creation.")

    def _split_cell(self, next_group_id: int) -> Optional['Cell']:
        """Splits the current cell into two distinct cells if septum formation is complete."""
        progress = min(1.0, self.septum_progress)
        self.division_site = len(self.PhysicsRepresentation.segments) // 2 # Dynamically determine the division site but TODO it has some instability when the segments are being added during division

        if self._mother_septum_segments is None and self._daughter_septum_segments is None:
            self._mother_septum_segments = []
            self._daughter_septum_segments = []
            for i in range(self.num_septum_segments):
                mother_idx = self.division_site - 1 - i
                daughter_idx = self.division_site + i
                if mother_idx >= 0 and daughter_idx < len(self.PhysicsRepresentation.segments):
                    # Recreate shape and update the segment's shape reference
                    self._mother_septum_segments.append(self.PhysicsRepresentation.segments[mother_idx])
                    self._daughter_septum_segments.append(self.PhysicsRepresentation.segments[daughter_idx])

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

        current_length = len(self.PhysicsRepresentation.segments)
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
            space=self.space, config=self.config, start_pos=self.PhysicsRepresentation.segments[mother_final_len].position,
            group_id=next_group_id, _from_division=True, base_color=daughter_color
        )

        daughter_cell.PhysicsRepresentation.segments = self.PhysicsRepresentation.segments[mother_final_len:]
        for segment in daughter_cell.PhysicsRepresentation.segments:
            segment.shape.filter = pymunk.ShapeFilter(group=next_group_id)

        connecting_joint_idx = mother_final_len - 1

        # 1. Assign the daughter's joints
        daughter_cell.PhysicsRepresentation.pivot_joints = self.PhysicsRepresentation.pivot_joints[mother_final_len:]
        if self.config.ROTARY_LIMIT_JOINT:
            daughter_cell.PhysicsRepresentation.limit_joints = self.PhysicsRepresentation.limit_joints[mother_final_len:]
        if self.config.DAMPED_ROTARY_SPRING:
            daughter_cell.PhysicsRepresentation.spring_joints = self.PhysicsRepresentation.spring_joints[mother_final_len:]

        # 2. Remove the connecting joints from the physics space
        self.space.remove(self.PhysicsRepresentation.pivot_joints[connecting_joint_idx])
        if self.config.ROTARY_LIMIT_JOINT:
            self.space.remove(self.PhysicsRepresentation.limit_joints[connecting_joint_idx])
        if self.config.DAMPED_ROTARY_SPRING:
            self.space.remove(self.PhysicsRepresentation.spring_joints[connecting_joint_idx])

        # 3. Trim the mother's components
        self.PhysicsRepresentation.segments = self.PhysicsRepresentation.segments[:mother_final_len]
        self.PhysicsRepresentation.pivot_joints = self.PhysicsRepresentation.pivot_joints[:connecting_joint_idx]
        if self.config.ROTARY_LIMIT_JOINT:
            self.PhysicsRepresentation.limit_joints = self.PhysicsRepresentation.limit_joints[:connecting_joint_idx]
        if self.config.DAMPED_ROTARY_SPRING:
            self.PhysicsRepresentation.spring_joints = self.PhysicsRepresentation.spring_joints[:connecting_joint_idx]

        self._update_colors()
        daughter_cell._update_colors()

        daughter_cell.growth_accumulator_tail = self.PhysicsRepresentation.growth_accumulator_tail

        self.PhysicsRepresentation.growth_accumulator_tail = 0.0
        self.num_divisions += 1

        return daughter_cell

    def _update_colors(self) -> None:
        if not self.PhysicsRepresentation.segments: return
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
        for segment in self.PhysicsRepresentation.segments:
            segment.shape.color = body_color
        self.PhysicsRepresentation.segments[0].shape.color = head_color
        self.PhysicsRepresentation.segments[-1].shape.color = tail_color


