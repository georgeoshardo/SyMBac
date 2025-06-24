import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np
from typing import Optional, cast
from symbac.misc import generate_color
from symbac.simulation.config import CellConfig
from symbac.simulation.physics_representation import PhysicsRepresentation
from symbac.simulation.segments import CellSegment
import colorsys

from symbac.simulation.visualisation import ColonyVisualiser


# Note that length units here are the number of spheres in the cell, TODO: implement the continuous length measurement for rendering.
class Cell:
    _daughter_septum_segments: Optional[list[CellSegment]] = None
    _mother_septum_segments: Optional[list[CellSegment]] = None


    def __init__(
            self,
            space: pymunk.Space,
            config: CellConfig,
            start_pos: Vec2d,
            group_id: int = 0,
            _from_division: bool = False,
            base_color: Optional[tuple[int, int, int]] = None
    ) -> None:

        if isinstance(start_pos, tuple):
            start_pos = Vec2d(*start_pos)

        self.space = space
        self.config = config
        self.start_pos = start_pos
        self._group_id = group_id
        if not base_color:
            self.base_color = generate_color(group_id)
        else:
            self.base_color = base_color

        self.PhysicsRepresentation = PhysicsRepresentation(
            space=self.space,
            config=self.config,
            group_id=self._group_id,
            start_pos=self.start_pos,
            _from_division=_from_division,
        ) # Use dependency injection pattern for the physics representation, can't think of a better way

        self.is_dividing = False
        self._septum_progress = 0.0
        self.septum_duration = 1.5
        self.division_site: int | None = None
        self.num_septum_segments = self.config.GRANULARITY
        self.min_septum_radius = self.config.SEGMENT_RADIUS * 0.1
        self.length_at_division_start = 0
        self.division_bias = 0 #self.config.GRANULARITY # not needed when bidirectional growth I think

        variation = self.config.BASE_MAX_LENGTH * self.config.MAX_LENGTH_VARIATION
        random_max_len = np.random.uniform(
            self.config.BASE_MAX_LENGTH - variation, self.config.BASE_MAX_LENGTH + variation
        )

        self._max_length = max(self.config.MIN_LENGTH_AFTER_DIVISION * 2, int(random_max_len))

        self.adjusted_growth_rate = self.config.GROWTH_RATE

        self.num_divisions = 0
        self._mother_septum_segments = None
        self._daughter_septum_segments = None

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
        return len(self.PhysicsRepresentation.segments) // 2 # Dynamically determine the division site but

    @division_site.setter
    def division_site(self, value: int) -> None:
        """
        Setter for division site, ensures it is within the valid range.
        """
        self._division_site = value



