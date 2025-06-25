# colony.py
from typing import Iterable, Callable, Optional

from symbac.simulation.simcell import SimCell
from pymunk.space import Space


class Colony:
    def __init__(self, space: Space, cells: list[SimCell]) -> None:
        self.space = space
        self.next_group_id = 1
        self.cells: list[SimCell] = cells
        self.newly_born_cells_map: dict[int, list[SimCell]] = {}

    def __len__(self):
        return len(self.cells)

    def add_cell(self, cell: SimCell) -> None:
        self.cells.append(cell)

    def add_cells(self, cells: Iterable[SimCell]) -> None:
        self.cells.extend(cells)

    def handle_cell_overlaps(self, newly_born_cells_map: dict[SimCell, SimCell]) -> None:
        for daughter, mother in newly_born_cells_map.items():
            mother_shapes = [s.shape for s in mother.PhysicsRepresentation.segments]

            # Symmetrical Overlap Removal Loop
            while True:
                overlap_found = False
                # We only need to check the daughter's head against the mother's body
                if daughter.PhysicsRepresentation.segments:
                    daughter_head = daughter.PhysicsRepresentation.segments[0]
                    query_result = self.space.shape_query(daughter_head.shape)

                    for info in query_result:
                        # If the daughter's head is overlapping with the mother
                        if info.shape in mother_shapes:
                            mother.PhysicsRepresentation.remove_tail_segment()
                            daughter.PhysicsRepresentation.remove_head_segment()

                            # We must update the mother_shapes list since a shape was removed
                            mother_shapes.pop()  # TODO this could be an issue leading to an infinite loop if the mother has no segments left and the minimum length is too high

                            overlap_found = True
                            break  # Exit the inner query loop

                # If we went through a full check without finding an overlap, we're done.
                if not overlap_found:
                    break
