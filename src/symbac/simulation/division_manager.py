import typing
import pymunk
from symbac.simulation.simcell import SimCell
from symbac.simulation.config import CellConfig


class DivisionManager:

    def __init__(self, space: pymunk.Space, config: CellConfig) -> None:

        """
        Handles the division of a cell.
        Note that `symbac.simulation.SimCell` objects control the division manager.

        Args:
            space: The simulation space
            config: The cell's configuration
        """

        self.space = space
        self.config = config

    def ready_to_divide(self, cell: 'SimCell') -> bool:
        """A simple state check. Returns True if the cell is ready to divide."""
        if cell.is_dividing:
            return False

        if len(cell.physics_representation.segments) < cell.max_length: #LENGTH_FIX
            return False

        split_index = self.get_split_index(cell)
        if split_index < self.config.MIN_LENGTH_AFTER_DIVISION or \
                (cell.physics_representation.num_segments - split_index) < self.config.MIN_LENGTH_AFTER_DIVISION:

            return False

        return True

    def get_split_index(self, cell: 'SimCell') -> int:
        """
        Returns the index at which the cell should be split for division.
        This is typically the middle of the segments.
        """
        return cell.physics_representation.num_segments // 2

    def set_division_readiness(self, cell: 'SimCell', division_model = "sizer") -> None:
        if division_model == "sizer":
            cell.is_dividing = True
            cell.septum_progress = 0.0
            cell.division_site = self.get_split_index(cell)
            cell.length_at_division_start = len(cell.physics_representation.segments) #LENGTH_FIX
            return None

    def reset_division_readiness(self, cell: 'SimCell') -> None:
        cell.is_dividing = False
        cell.septum_progress = 0.0
        cell.division_site = None
        return None

    def update_septum_progress(self, cell: 'SimCell', dt: float) -> None:
        """
        Updates the septum progress during division.
        This is a placeholder for actual septum formation logic.
        """
        cell.septum_progress += dt / self.config.SEPTUM_DURATION


    def initialise_mother_daughter_septum_segments(self, cell: 'SimCell') -> None:
        """Initialises the septum segments for mother and daughter cells."""
        assert cell.physics_representation._mother_septum_segments is None and cell.physics_representation._daughter_septum_segments is None
        cell.physics_representation._mother_septum_segments = []
        cell.physics_representation._daughter_septum_segments = []
        for i in range(self.config.NUM_SEPTUM_SEGMENTS):
            mother_idx = cell.division_site - 1 - i
            daughter_idx = cell.division_site + i
            if mother_idx >= 0 and daughter_idx < cell.physics_representation.num_segments:
                # Recreate shape and update the segment's shape reference
                cell.physics_representation._mother_septum_segments.append(cell.physics_representation.segments[mother_idx])
                cell.physics_representation._daughter_septum_segments.append(cell.physics_representation.segments[daughter_idx])

    def update_septum_segment_radii(self, cell: 'SimCell') -> None:
        """
        Updates the radii of the septum segments based on the septum progress.
        This is a placeholder for actual septum formation logic.
        """
        assert cell.physics_representation._mother_septum_segments is not None and cell.physics_representation._daughter_septum_segments is not None
        for i in range(self.config.NUM_SEPTUM_SEGMENTS):
            falloff = (self.config.NUM_SEPTUM_SEGMENTS - i) / self.config.NUM_SEPTUM_SEGMENTS
            shrinkage = (self.config.SEGMENT_RADIUS - self.config.MIN_SEPTUM_RADIUS) * cell.septum_progress * falloff
            new_radius = self.config.SEGMENT_RADIUS - shrinkage

            # Recreate shape and update the segment's shape reference
            cell.physics_representation._mother_septum_segments[i].radius = new_radius
            cell.physics_representation._daughter_septum_segments[i].radius = new_radius

    def restore_segment_radii(self, cell: 'SimCell') -> None:
        """
        Restores the radii of the septum segments to their original values.
        This is called after division is complete.
        """
        # Restore original radius after split
        assert cell.physics_representation._mother_septum_segments is not None
        assert cell.physics_representation._daughter_septum_segments is not None
        for i in range(self.config.NUM_SEPTUM_SEGMENTS):
            cell.physics_representation._mother_septum_segments[i].radius = self.config.SEGMENT_RADIUS
            cell.physics_representation._daughter_septum_segments[i].radius = self.config.SEGMENT_RADIUS

    def split_cell(self, cell: 'SimCell', next_group_id: int) -> 'SimCell':


        current_length = len(cell.physics_representation.segments) #LENGTH_FIX
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

        cell.physics_representation._mother_septum_segments = None
        cell.physics_representation._daughter_septum_segments = None
        # --- START of MODIFIED SECTION for color inheritance ---


        daughter_cell = SimCell(space=self.space, config=self.config,
                                start_pos=cell.physics_representation.segments[mother_final_len].position,
                                group_id=next_group_id, _from_division=True)

        daughter_cell.physics_representation.segments = cell.physics_representation.segments[mother_final_len:]
        for segment in daughter_cell.physics_representation.segments:
            segment.shape.filter = pymunk.ShapeFilter(group=next_group_id)

        connecting_joint_idx = mother_final_len - 1

        # 1. Assign the daughter's joints
        daughter_cell.physics_representation.pivot_joints = cell.physics_representation.pivot_joints[mother_final_len:]
        if self.config.ROTARY_LIMIT_JOINT:
            daughter_cell.physics_representation.limit_joints = cell.physics_representation.limit_joints[mother_final_len:]
        if self.config.DAMPED_ROTARY_SPRING:
            daughter_cell.physics_representation.spring_joints = cell.physics_representation.spring_joints[mother_final_len:]

        # 2. Remove the connecting joints from the physics space
        self.space.remove(cell.physics_representation.pivot_joints[connecting_joint_idx])
        if self.config.ROTARY_LIMIT_JOINT:
            self.space.remove(cell.physics_representation.limit_joints[connecting_joint_idx])
        if self.config.DAMPED_ROTARY_SPRING:
            self.space.remove(cell.physics_representation.spring_joints[connecting_joint_idx])

        # 3. Trim the mother's components
        cell.physics_representation.segments = cell.physics_representation.segments[:mother_final_len]
        cell.physics_representation.pivot_joints = cell.physics_representation.pivot_joints[:connecting_joint_idx]
        if self.config.ROTARY_LIMIT_JOINT:
            cell.physics_representation.limit_joints = cell.physics_representation.limit_joints[:connecting_joint_idx]
        if self.config.DAMPED_ROTARY_SPRING:
            cell.physics_representation.spring_joints = cell.physics_representation.spring_joints[:connecting_joint_idx]

        cell.physics_representation.growth_accumulator_tail = 0.0
        cell.num_divisions += 1

        return daughter_cell

    def handle_division(self, cell: 'SimCell', next_group_id: int, dt: float) -> typing.Optional['SimCell']:
        if cell.is_dividing:  # If a cell is dividing
            self.update_septum_progress(cell, dt)  # Update the septum progress
            self.update_septum_segment_radii(cell)  # and Update the septum segment radii
            if cell.septum_progress < 1.0:  # If the septum is not fully formed
                new_cell = None  # Do nothing, just wait for the septum to form
            else:  # If the septum is fully formed
                self.restore_segment_radii(cell)  # Reset the segment radii to their original values ahead of splitting
                new_cell = self.split_cell(cell, next_group_id)  # split the cell and get a new cell
            if new_cell:  # If the division was completed
                self.reset_division_readiness(cell)  # and reset the division readiness of the mother cell
        else:  # If a cell is not dividing
            if self.ready_to_divide(cell):  # then check if it is ready to divide
                self.initialise_mother_daughter_septum_segments(cell)  # and if it is, initialise the septum segments
                self.set_division_readiness(cell)  # and if it is, set its state to dividing
            new_cell = None

        return new_cell