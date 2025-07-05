import typing
from pymunk.vec2d import Vec2d
import numpy as np
from symbac.simulation.config import CellConfig
from symbac.simulation.physics_representation import PhysicsRepresentation
if typing.TYPE_CHECKING:
    from pymunk.space import Space


# Note that length units here are the number of spheres in the cell, TODO: implement the continuous length measurement for rendering.
class SimCell:

    __slots__ = (
        "config",
        "_group_id",
        "physics_representation",
        "is_dividing",
        "_septum_progress",
        "_division_site",
        "num_segments_at_division_start",
        "division_bias",
        "max_length",
        "adjusted_growth_rate",
        "num_divisions",
        "birth_length"
        "length"
    )

    def __init__(
            self,
            space: 'Space',
            config: CellConfig,
            start_pos: Vec2d,
            group_id: int = 0,
            _from_division: bool = False,
    ) -> None:

        if isinstance(start_pos, tuple):
            start_pos = Vec2d(*start_pos)

        self.config = config
        self._group_id = group_id

        self.physics_representation = PhysicsRepresentation(
            space=space,
            config=self.config,
            group_id=self._group_id,
            start_pos=start_pos,
            _from_division=_from_division,
        ) # Use dependency injection pattern for the physics representation, can't think of a better way

        self.is_dividing = False
        self._septum_progress = 0.0
        self.division_site: int | None = None
        self.num_segments_at_division_start = 0 # NOTE:This is NOT the birth length, this is a variable to keep track of how much growth has occurred during septum formation and division
        self.division_bias = 0

        variation = self.config.BASE_MAX_LENGTH * self.config.MAX_LENGTH_VARIATION
        random_max_len = np.random.uniform(
            self.config.BASE_MAX_LENGTH - variation, self.config.BASE_MAX_LENGTH + variation
        ) # TODO: look at how to accurately model this distribution

        self.max_length = max(self.config.MIN_LENGTH_AFTER_DIVISION * 2, int(random_max_len))

        self.adjusted_growth_rate = self.config.GROWTH_RATE

        self.num_divisions = 0

    @property
    def length(self):
        if self.config.SIMPLE_LENGTH:
            return self.physics_representation.num_segments * self.config.JOINT_DISTANCE
        else:
            return self.physics_representation.get_continuous_length()

    @property
    def group_id(self) -> int:
        """Returns the group ID of the cell, but it's immutable now."""
        return self._group_id

    @group_id.setter
    def group_id(self, value: int) -> None:
        """
        Tell the user they shouldn't be trying to meddle with the group_id
        """
        raise AttributeError("Cell lineage ID (group_id) is immutable and cannot be changed after creation.")

    @property
    def septum_progress(self) -> float:
        return min(1.0, self._septum_progress)

    @septum_progress.setter
    def septum_progress(self, value: float) -> None:
        """
        Setter for septum progress, ensures it does not exceed 1.0.
        """
        if value < 0.0:
            raise ValueError("Septum progress cannot be negative.")
        self._septum_progress = value

    @property
    def division_site(self) -> int:
        return len(self.physics_representation.segments) // 2 # Dynamically determine the division site but

    @division_site.setter
    def division_site(self, value: int) -> None:
        """
        Setter for division site, ensures it is within the valid range.
        """
        self._division_site = value



