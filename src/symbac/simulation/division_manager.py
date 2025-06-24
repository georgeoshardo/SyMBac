import typing
import pymunk
from symbac.simulation.cell import Cell
from symbac.simulation.config import CellConfig


class DivisionManager:

    def __init__(self, space: pymunk.Space, config: CellConfig) -> None:
        self.space = space
        self.config = config

    def ready_to_divide(self, cell: 'Cell') -> bool:
        """A simple state check. Returns True if the cell is ready to divide."""
        if cell.is_dividing:
            return False

        if len(cell.PhysicsRepresentation.segments) < cell._max_length:
            return False

        split_index = self.get_split_index(cell)
        if split_index < self.config.MIN_LENGTH_AFTER_DIVISION or \
                (len(cell.PhysicsRepresentation.segments) - split_index) < self.config.MIN_LENGTH_AFTER_DIVISION:

            return False

        return True

    def get_split_index(self, cell: 'Cell') -> int:
        """
        Returns the index at which the cell should be split for division.
        This is typically the middle of the segments.
        """
        return len(cell.PhysicsRepresentation.segments) // 2

    def set_division_readiness(self, cell: 'Cell') -> None:
        cell.is_dividing = True
        cell.septum_progress = 0.0
        cell.division_site = self.get_split_index(cell)
        cell.length_at_division_start = len(cell.PhysicsRepresentation.segments)
        return None

    def reset_division_readiness(self, cell: 'Cell') -> None:
        cell.is_dividing = False
        cell.septum_progress = 0.0
        cell.division_site = None
        cell.length_at_division_start = 0
        return None

    def update_septum_progress(self, cell: 'Cell', dt: float) -> None:
        """
        Updates the septum progress during division.
        This is a placeholder for actual septum formation logic.
        """
        cell.septum_progress += dt / self.config.SEPTUM_DURATION


    def initialise_mother_daughter_septum_segments(self, cell: 'Cell') -> None:
        """Initialises the septum segments for mother and daughter cells."""
        assert cell.PhysicsRepresentation._mother_septum_segments is None and cell.PhysicsRepresentation._daughter_septum_segments is None
        cell.PhysicsRepresentation._mother_septum_segments = []
        cell.PhysicsRepresentation._daughter_septum_segments = []
        for i in range(self.config.NUM_SEPTUM_SEGMENTS):
            mother_idx = cell.division_site - 1 - i
            daughter_idx = cell.division_site + i
            if mother_idx >= 0 and daughter_idx < len(cell.PhysicsRepresentation.segments):
                # Recreate shape and update the segment's shape reference
                cell.PhysicsRepresentation._mother_septum_segments.append(cell.PhysicsRepresentation.segments[mother_idx])
                cell.PhysicsRepresentation._daughter_septum_segments.append(cell.PhysicsRepresentation.segments[daughter_idx])

    def update_septum_segment_radii(self, cell: 'Cell') -> None:
        """
        Updates the radii of the septum segments based on the septum progress.
        This is a placeholder for actual septum formation logic.
        """
        assert cell.PhysicsRepresentation._mother_septum_segments is not None and cell.PhysicsRepresentation._daughter_septum_segments is not None
        for i in range(self.config.NUM_SEPTUM_SEGMENTS):
            falloff = (self.config.NUM_SEPTUM_SEGMENTS - i) / self.config.NUM_SEPTUM_SEGMENTS
            shrinkage = (self.config.SEGMENT_RADIUS - self.config.MIN_SEPTUM_RADIUS) * cell.septum_progress * falloff
            new_radius = self.config.SEGMENT_RADIUS - shrinkage

            # Recreate shape and update the segment's shape reference
            cell.PhysicsRepresentation._mother_septum_segments[i].radius = new_radius
            cell.PhysicsRepresentation._daughter_septum_segments[i].radius = new_radius

    def restore_segment_radii(self, cell: 'Cell') -> None:
        """
        Restores the radii of the septum segments to their original values.
        This is called after division is complete.
        """
        # Restore original radius after split
        assert cell.PhysicsRepresentation._mother_septum_segments is not None
        assert cell.PhysicsRepresentation._daughter_septum_segments is not None
        for i in range(self.config.NUM_SEPTUM_SEGMENTS):
            cell.PhysicsRepresentation._mother_septum_segments[i].radius = self.config.SEGMENT_RADIUS
            cell.PhysicsRepresentation._daughter_septum_segments[i].radius = self.config.SEGMENT_RADIUS

    def split_cell(self, cell: 'Cell', next_group_id: int) -> 'Cell':


        current_length = len(cell.PhysicsRepresentation.segments)
        growth_during_division = current_length - cell.length_at_division_start
        original_half_length = cell.length_at_division_start // 2
        growth_to_redistribute_to_mother = growth_during_division // 2
        mother_final_len = original_half_length + growth_to_redistribute_to_mother
        mother_final_len += cell.division_bias
        daughter_length = current_length - mother_final_len

        if mother_final_len < self.config.MIN_LENGTH_AFTER_DIVISION:
            mother_final_len = self.config.MIN_LENGTH_AFTER_DIVISION
        elif daughter_length < self.config.MIN_LENGTH_AFTER_DIVISION:
            mother_final_len = current_length - self.config.MIN_LENGTH_AFTER_DIVISION
        elif mother_final_len >= current_length:
            mother_final_len = current_length - self.config.MIN_LENGTH_AFTER_DIVISION

        cell.PhysicsRepresentation._mother_septum_segments = None
        cell.PhysicsRepresentation._daughter_septum_segments = None
        # --- START of MODIFIED SECTION for color inheritance ---


        daughter_cell = Cell(
            space=self.space, config=self.config, start_pos=cell.PhysicsRepresentation.segments[mother_final_len].position,
            group_id=next_group_id, _from_division=True, base_color=None
        )

        daughter_cell.PhysicsRepresentation.segments = cell.PhysicsRepresentation.segments[mother_final_len:]
        for segment in daughter_cell.PhysicsRepresentation.segments:
            segment.shape.filter = pymunk.ShapeFilter(group=next_group_id)

        connecting_joint_idx = mother_final_len - 1

        # 1. Assign the daughter's joints
        daughter_cell.PhysicsRepresentation.pivot_joints = cell.PhysicsRepresentation.pivot_joints[mother_final_len:]
        if self.config.ROTARY_LIMIT_JOINT:
            daughter_cell.PhysicsRepresentation.limit_joints = cell.PhysicsRepresentation.limit_joints[mother_final_len:]
        if self.config.DAMPED_ROTARY_SPRING:
            daughter_cell.PhysicsRepresentation.spring_joints = cell.PhysicsRepresentation.spring_joints[mother_final_len:]

        # 2. Remove the connecting joints from the physics space
        self.space.remove(cell.PhysicsRepresentation.pivot_joints[connecting_joint_idx])
        if self.config.ROTARY_LIMIT_JOINT:
            self.space.remove(cell.PhysicsRepresentation.limit_joints[connecting_joint_idx])
        if self.config.DAMPED_ROTARY_SPRING:
            self.space.remove(cell.PhysicsRepresentation.spring_joints[connecting_joint_idx])

        # 3. Trim the mother's components
        cell.PhysicsRepresentation.segments = cell.PhysicsRepresentation.segments[:mother_final_len]
        cell.PhysicsRepresentation.pivot_joints = cell.PhysicsRepresentation.pivot_joints[:connecting_joint_idx]
        if self.config.ROTARY_LIMIT_JOINT:
            cell.PhysicsRepresentation.limit_joints = cell.PhysicsRepresentation.limit_joints[:connecting_joint_idx]
        if self.config.DAMPED_ROTARY_SPRING:
            cell.PhysicsRepresentation.spring_joints = cell.PhysicsRepresentation.spring_joints[:connecting_joint_idx]



        daughter_cell.growth_accumulator_tail = cell.PhysicsRepresentation.growth_accumulator_tail

        cell.PhysicsRepresentation.growth_accumulator_tail = 0.0
        cell.num_divisions += 1

        return daughter_cell