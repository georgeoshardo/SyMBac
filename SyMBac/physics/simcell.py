import typing
from pymunk.vec2d import Vec2d
import numpy as np
from SyMBac.physics.config import CellConfig
from SyMBac.physics.physics_representation import PhysicsRepresentation
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
        "birth_length",
        "current_segment_radius",
        "target_segment_radius",
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

        self.current_segment_radius = self.sample_segment_radius()
        self.target_segment_radius = self.current_segment_radius
        self.apply_current_width_to_segments()

        self.birth_length = self.length
        self.max_length = self.sample_max_length()

        self.adjusted_growth_rate = self.config.GROWTH_RATE

        self.num_divisions = 0

    def sample_max_length(self) -> float:
        sampled_max_length = np.random.normal(
            self.config.BASE_MAX_LENGTH, self.config.MAX_LENGTH_STD
        )
        min_length = max(
            1.0, self.config.MIN_LENGTH_AFTER_DIVISION * self.config.JOINT_DISTANCE * 2
        )
        # Ensure max_length > birth_length so the cell always has room to
        # grow before dividing.  Without this, a large MAX_LENGTH_STD can
        # produce a max_length below the cell's current length, creating a
        # deadlock where the cell can neither grow nor divide.
        # (Matches the legacy engine's ``max(daughter_length + 0.1, ...)``.)
        try:
            min_length = max(min_length, self.birth_length + self.config.JOINT_DISTANCE)
        except AttributeError:
            # birth_length is not yet set during __init__; the config-based
            # floor is sufficient for seed cells.
            pass
        return max(min_length, float(sampled_max_length))

    def sample_segment_radius(self) -> float:
        base_cell_width = 2.0 * self.config.SEGMENT_RADIUS
        sampled_width = np.random.normal(base_cell_width, self.config.WIDTH_STD)
        min_radius = max(0.1, self.config.SEGMENT_RADIUS * 0.3)
        radius = max(min_radius, float(sampled_width) / 2.0)
        if self.config.WIDTH_UPPER_LIMIT is not None:
            radius = min(radius, self.config.WIDTH_UPPER_LIMIT)
        return radius

    def set_new_width_target(self) -> None:
        self.target_segment_radius = self.sample_segment_radius()

    def apply_current_width_to_segments(self) -> None:
        if not self.physics_representation.segments:
            return
        for segment in self.physics_representation.segments:
            segment.set_radius(self.current_segment_radius)

    def sync_width_from_segments(self) -> None:
        if not self.physics_representation.segments:
            return
        avg_radius = float(np.mean([s.radius for s in self.physics_representation.segments]))
        self.current_segment_radius = avg_radius

    def update_width_transition(self, dt: float) -> None:
        if self.config.WIDTH_RELAXATION_TIME <= 0:
            self.current_segment_radius = self.target_segment_radius
            self.apply_current_width_to_segments()
            return

        alpha = 1.0 - np.exp(-dt / self.config.WIDTH_RELAXATION_TIME)
        self.current_segment_radius += (self.target_segment_radius - self.current_segment_radius) * alpha
        self.apply_current_width_to_segments()

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
        if self._division_site is not None:
            return self._division_site
        return len(self.physics_representation.segments) // 2

    @division_site.setter
    def division_site(self, value: int | None) -> None:
        """
        Setter for division site, ensures it is within the valid range.
        """
        self._division_site = value
