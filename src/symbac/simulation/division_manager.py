import typing
if typing.TYPE_CHECKING:
    from symbac.simulation.cell import Cell

class DivisionManager:

    def ready_to_divide(self, cell: 'Cell') -> bool:
        """A simple state check. Returns True if the cell is ready to divide."""
        if cell.is_dividing:
            return False

        if len(cell.PhysicsRepresentation.segments) < cell._max_length:
            return False

        split_index = self.get_split_index(cell)
        if split_index < cell.config.MIN_LENGTH_AFTER_DIVISION or \
                (len(cell.PhysicsRepresentation.segments) - split_index) < cell.config.MIN_LENGTH_AFTER_DIVISION:

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

    def update_septum(self, cell: 'Cell', dt: float) -> None:
        """
        Updates the septum progress during division.
        This is a placeholder for actual septum formation logic.
        """
        cell.septum_progress += dt / cell.septum_duration


    def initialise_mother_daughter_septum_segments(self, cell: 'Cell') -> None:
        """Initialises the septum segments for mother and daughter cells."""
        assert cell._mother_septum_segments is None and cell._daughter_septum_segments is None
        cell._mother_septum_segments = []
        cell._daughter_septum_segments = []
        for i in range(cell.num_septum_segments):
            mother_idx = cell.division_site - 1 - i
            daughter_idx = cell.division_site + i
            if mother_idx >= 0 and daughter_idx < len(cell.PhysicsRepresentation.segments):
                # Recreate shape and update the segment's shape reference
                cell._mother_septum_segments.append(cell.PhysicsRepresentation.segments[mother_idx])
                cell._daughter_septum_segments.append(cell.PhysicsRepresentation.segments[daughter_idx])
