import pymunk
import pymunk.pygame_util
from symbac.simulation.config import CellConfig


class CellSegment:
    """
    Represents a cell segment, encapsulating both the physical body and shape used in the Pymunk physics simulation.

    Attributes
    ----------
    config : CellConfig
        Configuration object containing physical parameters for the cell segment, such as mass and radius.
    group_id : int
        Group identifier to distinguish this cell segment from others in the physics simulation for collision filtering.
    body : pymunk.Body
        The physical body of the cell segment used by Pymunk, including its mass and moment of inertia.
    shape : pymunk.Circle
        The shape of the cell segment for collision detection and physical interactions in the simulation; defined as a circle with a given radius.
    angle : float
        The current rotation angle of the cell segment in radians.
    position : tuple[float, float]
        The current position of the cell segment in the simulation space.
    """
    def __init__(
            self,
            config: CellConfig,
            group_id: int,
            position: tuple[float, float] = (0.0, 0.0),
            angle: float = 0.0
    ) -> None:
        """
        Parameters
        ----------
        config : CellConfig
            Configuration object containing physical parameters for the cell segment,
            such as mass and radius.
        group_id : int
            Identifier for collision filtering. Shapes with the same group ID will
            not collide with each other.
        position : tuple[float, float], optional
            The initial position of the cell segment in the simulation space.
            Defaults to (0.0, 0.0).
        angle : float, optional
            The initial rotation angle of the cell segment in radians.
            Defaults to 0.0.
        """
        self.config = config
        self.group_id = group_id

        moment = pymunk.moment_for_circle(
            self.config.SEGMENT_MASS,
            0,
            self.config.SEGMENT_RADIUS
        )
        self.body = pymunk.Body(self.config.SEGMENT_MASS, moment)

        self.shape = pymunk.Circle(self.body, self.config.SEGMENT_RADIUS)
        self.shape.friction = 0.0
        self.shape.filter = pymunk.ShapeFilter(group=self.group_id)
        self.angle = angle
        self.position = position

    @property
    def position(self) -> tuple[float, float]:
        """Returns the current position of the segment."""
        return self.body.position[0], self.body.position[1]  # To return a tuple instead of Vec2d

    @position.setter
    def position(self, position: tuple[float, float]) -> None:
        self.body.position = position

    @property
    def angle(self) -> float:
        """Returns the current angle of the segment."""
        return self.body.angle

    @angle.setter
    def angle(self, angle: float) -> None:
        self.body.angle = angle


