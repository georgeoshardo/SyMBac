# colony.py
from typing import Iterable, Callable, Optional

from symbac.simulation.division_manager import DivisionManager
from symbac.simulation.growth_manager import GrowthManager
from symbac.simulation.simcell import SimCell
from symbac.simulation.visualisation.colony_visualiser import ColonyVisualiser
from pymunk.space import Space
import typing
if typing.TYPE_CHECKING:
    from symbac.simulation.sim_loop import SimulationContext


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

    def update(self, dt: float, division_manager: DivisionManager, growth_manager: GrowthManager, simulation_context: 'SimulationContext', cell_hook: Callable[['SimCell', 'SimulationContext'], None]) -> None:
        """
        Update the colony by processing each cell for growth and division.
        """
        for cell in self.cells:

            if cell_hook:
                cell_hook(cell, simulation_context)

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
                            ColonyVisualiser.update_colors(daughter)
                            ColonyVisualiser.update_colors(mother)

                            # We must update the mother_shapes list since a shape was removed
                            mother_shapes.pop()  # TODO this could be an issue leading to an infinite loop if the mother has no segments left and the minimum length is too high

                            overlap_found = True
                            break  # Exit the inner query loop

                # If we went through a full check without finding an overlap, we're done.
                if not overlap_found:
                    break
