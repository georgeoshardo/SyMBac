from dataclasses import dataclass
from typing import Optional, List, Callable
import inspect

import pymunk

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
    """Manages the main simulation loop, including physics, cell growth, and division.

    The Simulator class is the central orchestrator of the simulation. It
    initializes the Pymunk physics space, creates the initial cell colony,
    and manages the step-by-step execution of the simulation. It provides a
    flexible hooking mechanism to allow for custom logic to be injected at
    various points in the simulation cycle, such as before/after cell
    growth, after cell division, and at the end of each simulation step.


    Attributes:
        space: The Pymunk physics space instance.
        dt: The time step for the physics simulation. Units are arbitrary,
            depending on the cell's growth rate.
        colony: The colony of cells being simulated.
        growth_manager: Manages the growth of individual cells.
        division_manager: Manages the division of cells.
        next_group_id: The next available ID for new cells, ensuring unique
            collision filtering groups for new cells.
        frame_count: The number of simulation steps that have been executed.
        max_impulse_this_frame: Tracks the maximum impulse applied in the
            current physics frame.
        max_impulse_this_step: Tracks the maximum impulse applied over the
            entire simulation step (potentially multiple physics frames).
        max_joint_impulse: The maximum impulse observed on any joint, used
            for adaptive iteration adjustments.
        joint_impulse_threshold: A threshold for joint impulse, influencing
            adaptive physics iterations.
        adaptive_iterations: Flag to enable adaptive physics solver iterations.
            Experimental, may actually slow things down.
        pre_cell_grow_hooks: A list of callable functions to be executed on
            each cell before its growth is processed.
        post_cell_grow_hooks: A list of callable functions to be executed on
            each cell after its growth is processed.
        post_division_hooks: A list of callable functions to be executed after
            a cell divides, receiving both the mother and daughter cells as arguments.
        post_step_hooks: A list of callable functions to be executed at the end
            of each simulation step.
        post_init_hooks: A list of callable functions to be executed after the
            simulator has completed its initialization.
    """

    def __init__(
            self,
            physics_config: PhysicsConfig,
            initial_cell_config: CellConfig,
            post_init_hooks: Optional[List[Callable[['Simulator'], None]]] = None,
            pre_cell_grow_hooks: Optional[List[Callable[['SimCell'], None]]] = None,
            post_cell_grow_hooks: Optional[List[Callable[['SimCell'], None]]] = None,
            post_division_hooks: Optional[List[Callable[['SimCell', 'SimCell'], None]]] = None,
            post_step_hooks: Optional[List[Callable[['Simulator'], None]]] = None,
            adaptive_iterations: bool = False,
    ) -> None: #TODO allow a list of initial cells to be passed with their corresponding configs to set up a colony
        """Initializes the Simulator.

        Args:
            physics_config: Configuration for the Pymunk physics space,
                including gravity, damping, and iterations.
            initial_cell_config: Configuration for the first cell to be
                created in the simulation, including its starting position
                and physical properties.
            post_init_hooks: A list of hooks to be executed after the simulator
                is fully initialized. Each hook should accept a [`Simulator`][symbac.simulation.simulator.Simulator]
                instance as its sole argument.
            pre_cell_grow_hooks: A list of hooks to be executed on each cell
                before it grows. Each hook should accept a [`Simcell`][symbac.simulation.simcell.SimCell] instance
                as its sole argument.
            post_cell_grow_hooks: A list of hooks to be executed on each cell
                after it grows. Each hook should accept a [`Simcell`][symbac.simulation.simcell.SimCell] instance
                as its sole argument.
            post_division_hooks: A list of hooks to be executed after a cell
                divides. Each hook should accept the mother [`Simcell`][symbac.simulation.simcell.SimCell] and the
                newly created daughter [`Simcell`][symbac.simulation.simcell.SimCell] as arguments.
            post_step_hooks: A list of hooks to be executed at the end of
                each simulation step. Each hook should accept a [`Simulator`][symbac.simulation.simulator.Simulator]
                instance as its sole argument.
            adaptive_iterations: If True, the number of Pymunk solver iterations
                will be adjusted dynamically based on joint impulses. This is an
                experimental feature and may potentially slow down the simulation.

        Note:
            Currently, the simulator initializes with a single cell. Future
            versions may allow passing a list of initial cells and their
            corresponding configurations to set up a diverse colony from the start.
        """


        space = pymunk.Space(threaded=physics_config.THREADED)
        self.space: pymunk.Space = space
        self.space.threads = physics_config.THREADS
        self.space.iterations = physics_config.ITERATIONS
        self.space.gravity = physics_config.GRAVITY
        self.space.damping = physics_config.DAMPING
        self.dt: float = physics_config.DT
        if physics_config.COLLISION_SLOP:
            self.space.collision_slop = physics_config.COLLISION_SLOP

        self.next_group_id: int = 1
        initial_cell = SimCell(self.space, config=initial_cell_config, start_pos=initial_cell_config.START_POS,
                               group_id=self.next_group_id)
        self.colony: Colony = Colony(self.space, [initial_cell])
        self.next_group_id += 1
        self.growth_manager: GrowthManager = GrowthManager()
        self.division_manager: DivisionManager = DivisionManager(self.space, initial_cell_config)

        self.frame_count: int = 0


        self.max_impulse_this_frame: float = 0.0

        self.max_impulse_this_step: float = 0.0
        self.max_joint_impulse: float = 0.0
        self.joint_impulse_threshold: float = 100.
        self.adaptive_iterations: bool = adaptive_iterations

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

        self.post_init_hooks: List[Callable[['Simulator'], None]] = []
        if post_init_hooks:
            for hook in post_init_hooks:
                self.add_and_run_post_init_hook(hook) #This also runs the hook


    @staticmethod
    def _validate_hook_signature(hook: Callable, expected_params: int):
        """Validates that a hook function has the expected number of parameters.

        Args:
            hook: The hook function to validate.
            expected_params: The expected number of positional parameters
                the hook function should accept.

        Raises:
            ValueError: If the hook has an invalid number of parameters.
            TypeError: If the hook signature cannot be inspected (e.g., if
                it's not a valid callable).
        """
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
        """Registers a hook to be called after each simulation step.


        Args:
            hook: The function to be called. It must accept a single
                [`Simulator`][symbac.simulation.simulator.Simulator] instance as its argument.

        Returns:
            None

        Examples:
            A common use case is a function that removes cells if they move
            beyond a critical position, which is useful for simulations
            involving microfluidic geometries to discard cells that exit
            the defined domain.
            >>> def cell_remover(simulator: 'Simulator') -> None:
            ...     for cell in simulator.cells:
            ...         if cell.physics_representation.segments[0].body.position.y > 1000:
            ...             simulator.colony.delete_cell(cell)
        """
        self._validate_hook_signature(hook, expected_params=1)
        self.post_step_hooks.append(hook)

    def add_pre_cell_grow_hook(self, hook: Callable[['SimCell'], None]) -> None:
        """Registers a hook to be called on a cell before it grows.

        Args:
            hook: The function to be called. It must
                accept a single [`SimCell`][symbac.simulation.simcell.SimCell] instance as its argument.

        Returns:
            None

        Examples:
            A function which calculates the cell's compression ratio, and updates
                its growth rate to slow it down if under compression.

            >>> import numpy as np
            >>> def cell_growth_rate_updater(cell: SimCell) -> None:
            ...     compression_ratio = cell.physics_representation.get_compression_ratio()
            ...     cell.adjusted_growth_rate = cell.config.GROWTH_RATE * compression_ratio**4
            ...
            ...     variation = cell.config.BASE_MAX_LENGTH * cell.config.MAX_LENGTH_VARIATION
            ...     random_max_len = np.random.uniform(
            ...     cell.config.BASE_MAX_LENGTH - variation, cell.config.BASE_MAX_LENGTH + variation
            ...     ) * np.sqrt(compression_ratio)
            ...
            ...     cell.max_length = max(cell.length, int(random_max_len))
        """
        self._validate_hook_signature(hook, expected_params=1)
        self.pre_cell_grow_hooks.append(hook)

    def add_post_cell_grow_hook(self, hook: Callable[['SimCell'], None]) -> None:
        """Registers a hook to be called on a cell after it grows.

        Args:
            hook: The function to be called. It must accept a single
                [`SimCell`][symbac.simulation.simcell.SimCell] instance as its argument.
        """
        self._validate_hook_signature(hook, expected_params=1)
        self.post_cell_grow_hooks.append(hook)

    def add_post_division_hook(self, hook: Callable[['SimCell', 'SimCell'], None]) -> None:
        """Registers a hook to be called after a cell divides.

        Args:
            hook: The function to be called. It must accept two [`SimCell`][symbac.simulation.simcell.SimCell]
                instances as arguments: the mother cell and the newly
                created daughter cell.
        """
        self._validate_hook_signature(hook, expected_params=2)
        self.post_division_hooks.append(hook)

    def add_and_run_post_init_hook(self, hook: Callable[['Simulator'], None]) -> None:
        """Registers a hook to be called after the simulator is initialized.

        The registered hook is executed immediately upon registration, allowing
        for immediate post-initialization setup or validation.

        Args:
            hook: The function to be called. It must accept a single
                [`Simulator`][symbac.simulation.simulator.Simulator] instance as its argument.
        """
        self._validate_hook_signature(hook, expected_params=1)
        self.post_init_hooks.append(hook)

        #Call the hook immediately after adding it in case it's added after instantiation
        hook(self)


    def step(self) -> None:
        """Executes a single simulation step.

        This method orchestrates the per-cell growth and division processes,
        updates the Pymunk physics space, and triggers all registered hooks
        at appropriate points in the simulation cycle. It also handles
        adding newly born cells to the colony and managing cell overlaps.
        If `adaptive_iterations` is enabled, it dynamically adjusts the
        Pymunk solver iterations based on observed joint impulses (but this is experimental and may make things slower).
        """

        newly_born_cells_map = {}

        # This is probably the best way to handle the simulation step without encapsulating and hiding too much logic into the Colony
        for cell in self.colony.cells[:]:
            # --- Pre cell hook
            for hook in self.pre_cell_grow_hooks:
                hook(cell)

            self.growth_manager.grow(cell, self.dt)  # Grow the cell

            for hook in self.post_cell_grow_hooks:
                hook(cell)
            new_cell: Optional['SimCell'] = self.division_manager.handle_division(cell, self.next_group_id,
                                                                             self.dt)  # Handle the cell division
            if new_cell is not None:  # If a new cell was created
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
        """Returns a list of all cells currently managed by the colony.

        This is a convenience property providing direct access to the
        [`Colony`][symbac.simulation.colony.Colony]'s internal list of [`SimCell`][symbac.simulation.simulator.Simulator] objects.
        """
        return self.colony.cells

    @property
    def num_cells(self) -> int:
        """Returns the total number of cells currently in the colony.

        This is a convenience property providing the current count of [`SimCell`][symbac.simulation.simulator.Simulator]
        objects managed by the [`Colony`][symbac.simulation.colony.Colony].
        """
        return len(self.colony)

