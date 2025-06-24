from dataclasses import dataclass
from typing import Optional

import pymunk
from pymunk import Vec2d

from symbac.simulation.colony import Colony
from symbac.simulation.config import PhysicsConfig, CellConfig
from symbac.simulation.division_manager import DivisionManager
from symbac.simulation.growth_manager import GrowthManager
from symbac.simulation.simcell import SimCell
from symbac.simulation.visualisation.colony_visualiser import ColonyVisualiser


@dataclass
class SimulationContext:
    """Holds shared data for a single simulation step."""
    frame_count: int
    dt: float

class Simulator:
    def __init__(
            self,
            physics_config: PhysicsConfig,
            initial_cell_config: CellConfig,
            post_division_hook: Optional[callable] = None,
            post_cell_iter_hook: Optional[callable] = None,
    ) -> None: #TODO allow a list of initial cells to be passed with their corresponding configs to set up a colony


        space = pymunk.Space(threaded=PhysicsConfig.THREADED)
        self.space = space
        self.space.threads = physics_config.THREADS
        self.space.iterations = physics_config.ITERATIONS
        self.space.gravity = physics_config.GRAVITY
        self.space.damping = physics_config.DAMPING
        self.dt = physics_config.DT


        self.next_group_id = 1
        initial_cell = SimCell(
            self.space,
            config=initial_cell_config,
            start_pos=Vec2d(0, 0),
            group_id=self.next_group_id,
        )
        self.colony = Colony(self.space, [initial_cell])
        self.next_group_id += 1
        self.growth_manager = GrowthManager()
        self.division_manager = DivisionManager(self.space, initial_cell_config)

        self.frame_count = 0

        self.post_division_hook = post_division_hook
        self.post_cell_iter_hook = post_cell_iter_hook

    def step(self):

        newly_born_cells_map = {}

        # This is probably the best way to handle the simulation step without encapsulating and hiding too much logic into the Colony
        for cell in self.colony.cells[:]:
            # cell_hook(cell, simulation_context)
            self.growth_manager.grow(cell, self.dt)  # Grow the cell
            ColonyVisualiser.update_colors(cell)  # and handle the colour update of the current cell

            new_cell: Optional['SimCell'] = self.division_manager.handle_division(cell, self.next_group_id,
                                                                             self.dt)  # Handle the cell division
            if new_cell is not None:  # If a new cell was created
                new_cell.base_color = ColonyVisualiser.get_daughter_colour(cell,
                                                                           self.next_group_id)  # and set the daughter's base colour for the visualisation
                ColonyVisualiser.update_colors(new_cell)  # update the colours of the cell according to rules
                newly_born_cells_map[new_cell] = cell  # Add the new cell to the map
                self.next_group_id += 1  # and increment the group ID

        # --- Handle adding newly born cells to the colony ---
        if newly_born_cells_map:
            self.colony.add_cells(newly_born_cells_map.keys())
            self.colony.handle_cell_overlaps(newly_born_cells_map)

        self.space.step(self.dt)
        self.frame_count += 1
        print(len(self.colony.cells), "cells in the colony after step", self.frame_count)

