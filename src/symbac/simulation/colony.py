# colony.py
from typing import Iterable, Callable, Optional

from symbac.simulation.simcell import SimCell
from pymunk.space import Space
import pymunk

class Colony:
    def __init__(self, space: Space, cells: list[SimCell]) -> None:
        """
        Create a colony object

        Args:
            space: The simulation space
            cells: The list of cells to initialise the colony with
        """
        self.space = space
        self.next_group_id = 1
        self.cells: list[SimCell] = cells
        self.newly_born_cells_map: dict[int, list[SimCell]] = {}

    def __len__(self):
        return len(self.cells)

    def add_cell(self, cell: SimCell) -> None:
        """
        Add a cell to the colony

        Args:
            cell: The SimCell object to be added.
        """
        self.cells.append(cell)

    def add_cells(self, cells: Iterable[SimCell]) -> None:
        """
        Add multiple cells to the colony

        Args:
            cells: The SimCell objects to be added.
        """
        self.cells.extend(cells)

    def handle_cell_overlaps(self, newly_born_cells_map: dict[SimCell, SimCell]) -> None:
        """
        Cells in SyMBac are composed of overlapping circles connecetd by joints. When a cell divides in two, the circles at the new poles the newly created cells must be removed to prevent the cells from overlapping
        This method handles that. It is very unlikely you'll need to manually invoke this.

        Args:
            newly_born_cells_map: A dictionary which maps a mother cell to its daugher cell, created during the simulation loop
        """
        for daughter, mother in newly_born_cells_map.items():
            mother_shapes = [s.shape for s in mother.physics_representation.segments]

            # Symmetrical Overlap Removal Loop
            while True:
                overlap_found = False
                # We only need to check the daughter's head against the mother's body
                if daughter.physics_representation.segments:
                    daughter_head = daughter.physics_representation.segments[0]
                    query_result = self.space.shape_query(daughter_head.shape)

                    for info in query_result:
                        # If the daughter's head is overlapping with the mother
                        if info.shape in mother_shapes:

                            mother.physics_representation.remove_tail_segment()
                            daughter.physics_representation.remove_head_segment()

                            # We must update the mother_shapes list since a shape was removed
                            mother_shapes.pop()  # TODO this could be an issue leading to an infinite loop if the mother has no segments left and the minimum length is too high

                            overlap_found = True
                            break  # Exit the inner query loop

                # If we went through a full check without finding an overlap, we're done.
                if not overlap_found:
                    break

# In colony.py, within the Colony class

    def delete_cell(self, cell: SimCell) -> None:
        """
        Remove a cell and all its associated physics objects from the simulation.

        Args:
            cell: The SimCell object to be removed.
        """
        if cell not in self.cells:
            # Cell might have already been removed, so we can just return.
            return

        # 1. Remove all physics objects from the space
        phys_rep = cell.physics_representation
        objs_to_remove = phys_rep.pivot_joints + phys_rep.limit_joints + phys_rep.spring_joints + [segment.shape for segment in phys_rep.segments] + [segment.body for segment in phys_rep.segments]

        self.space.remove(*objs_to_remove) # Remove all joints and segments in one go, it's faster

        # 2. Remove the cell from the colony's list
        self.cells.remove(cell)