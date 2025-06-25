from dataclasses import dataclass
from typing import Optional, List, Callable
import inspect
import pymunk
from pymunk import Vec2d

from symbac.simulation.colony import Colony
from symbac.simulation.config import PhysicsConfig, CellConfig
from symbac.simulation.division_manager import DivisionManager
from symbac.simulation.growth_manager import GrowthManager
from symbac.simulation.simcell import SimCell


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
            pre_cell_grow_hooks: Optional[List[Callable[['SimCell'], None]]] = None,
            post_cell_grow_hooks: Optional[List[Callable[['SimCell'], None]]] = None,
            post_division_hooks: Optional[List[Callable[['SimCell', 'SimCell'], None]]] = None,
            post_step_hooks: Optional[List[Callable[['Simulator'], None]]] = None,
            adaptive_iterations: bool = False,
    ) -> None: #TODO allow a list of initial cells to be passed with their corresponding configs to set up a colony


        space = pymunk.Space(threaded=physics_config.THREADED)
        self.space = space
        self.space.threads = physics_config.THREADS
        self.space.iterations = physics_config.ITERATIONS
        self.space.gravity = physics_config.GRAVITY
        self.space.damping = physics_config.DAMPING
        self.dt = physics_config.DT
        if physics_config.COLLISION_SLOP:
            self.space.collision_slop = physics_config.COLLISION_SLOP

        self.next_group_id = 1
        initial_cell = SimCell(
            self.space,
            config=initial_cell_config,
            start_pos=initial_cell_config.START_POS,
            group_id=self.next_group_id,
        )
        self.colony = Colony(self.space, [initial_cell])
        self.next_group_id += 1
        self.growth_manager = GrowthManager()
        self.division_manager = DivisionManager(self.space, initial_cell_config)

        self.frame_count = 0


        self.max_impulse_this_frame = 0.0
        self.substeps = 1 # Start with 1 sub-step

        self.max_impulse_this_step = 0.0
        self.max_joint_impulse = 0.0
        self.joint_impulse_threshold = 100
        self.adaptive_iterations = adaptive_iterations

        self.pre_cell_grow_hooks: List[Callable[['SimCell'], None]] = []
        if pre_cell_grow_hooks:
            for hook in pre_cell_grow_hooks:
                self.add_pre_cell_grow_hook(hook)

        self.post_cell_grow_hooks: List[Callable[['SimCell'], None]] = []
        if post_cell_grow_hooks:
            for hook in post_cell_grow_hooks:
                self.add_post_cell_grow_hook(hook) # Use the registration method to validate

        self.post_division_hooks: List[Callable[['SimCell', 'SimCell'], None]] = []
        if post_division_hooks:
            for hook in post_division_hooks:
                self.add_post_division_hook(hook)

        self.post_step_hooks: List[Callable[['Simulator'], None]] = []
        if post_step_hooks:
            for hook in post_step_hooks:
                self.add_post_step_hook(hook) # Use the registration method to validate



    @staticmethod
    def _validate_hook_signature(hook: Callable, expected_params: int):
        """Validates that a hook function has the expected number of parameters."""
        try:
            sig = inspect.signature(hook)
            if len(sig.parameters) != expected_params:
                raise ValueError(
                    f"Hook '{hook.__name__}' has an invalid signature. "
                    f"Expected {expected_params} parameters, but it has {len(sig.parameters)}."
                )
        except TypeError as e:
            raise TypeError(f"Could not inspect signature of hook {hook}. Is it a valid callable?") from e

    def add_post_step_hook(self, hook: Callable[['Simulator'], None]) -> None:
        """Registers a hook to be called after each simulation step."""
        self._validate_hook_signature(hook, expected_params=1)
        self.post_step_hooks.append(hook)

    def add_pre_cell_grow_hook(self, hook: Callable[['SimCell'], None]) -> None:
        self._validate_hook_signature(hook, expected_params=1)
        self.pre_cell_grow_hooks.append(hook)

    def add_post_cell_grow_hook(self, hook: Callable[['SimCell'], None]) -> None:
        self._validate_hook_signature(hook, expected_params=1)
        self.post_cell_grow_hooks.append(hook)

    def add_post_division_hook(self, hook: Callable[['SimCell', 'SimCell'], None]) -> None:
        self._validate_hook_signature(hook, expected_params=2)
        self.post_division_hooks.append(hook)

    def step(self):

        newly_born_cells_map = {}

        # This is probably the best way to handle the simulation step without encapsulating and hiding too much logic into the Colony
        for cell in self.colony.cells[:]:
            for hook in self.pre_cell_grow_hooks:
                hook(cell)

            self.growth_manager.grow(cell, self.dt)  # Grow the cell

            for hook in self.post_cell_grow_hooks:
                hook(cell)
            new_cell: Optional['SimCell'] = self.division_manager.handle_division(cell, self.next_group_id,
                                                                             self.dt)  # Handle the cell division
            if new_cell is not None:  # If a new cell was created
                #new_cell.base_color = ColonyVisualiser.get_daughter_colour(cell,
                #                                                           self.next_group_id)  # and set the daughter's base colour for the visualisation
                #ColonyVisualiser.update_colors(new_cell)  # update the colours of the cell according to rules
                newly_born_cells_map[new_cell] = cell  # Add the new cell to the map
                self.next_group_id += 1  # and increment the group ID
                # --- Post division hooks ---
                for hook in self.post_division_hooks:
                    hook(cell, new_cell)

        # --- Handle adding newly born cells to the colony ---
        if newly_born_cells_map:
            self.colony.add_cells(newly_born_cells_map.keys())
            self.colony.handle_cell_overlaps(newly_born_cells_map)

        if self.adaptive_iterations:
            for constraint in self.space.constraints:
                # Every joint type is a subclass of Constraint and has the 'impulse' property.
                if constraint.impulse > self.max_joint_impulse:
                    self.max_joint_impulse = constraint.impulse
                    self.space.iterations = self.space.iterations +  1

                    print("Increased space iterations to", self.space.iterations, "due to joint impulse of", constraint.impulse)


        self.space.step(self.dt)
        # --- Post step hooks ---
        for hook in self.post_step_hooks:
            hook(self)

        if self.adaptive_iterations:
            self.space.iterations = max(10, int(self.space.iterations * 0.9))
            self.max_joint_impulse *= 0.9

        self.frame_count += 1

    @property
    def cells(self):
        """Convenience property which returns a list of all cells in the colony."""
        return self.colony.cells

    @property
    def num_cells(self) -> int:
        """Convenience property which returns the number of cells in the colony."""
        return len(self.colony)

