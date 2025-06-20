from cell import Cell
from segments import CellSegment
from typing import Optional

class CellDivision:
    """
    Class representing a cell division event in a simulation.
    """

    def __init__(self, cell: 'Cell'):

        self.cell = cell


    def remove_tail_segment(self) -> Optional[CellSegment]:
        """Safely removes the last segment of the cell."""
        if len(self.cell.segments) <= self.cell.config.MIN_LENGTH_AFTER_DIVISION:
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