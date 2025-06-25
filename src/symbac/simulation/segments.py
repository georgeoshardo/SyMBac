import pymunk
from  symbac.simulation.config import CellConfig
from typing import Optional
import typing
if typing.TYPE_CHECKING:
    from pymunk.body import Body
    from pymunk.shapes import Circle
    from pymunk.space import Space

class CellSegment:
    config: CellConfig
    group_id: int
    body: 'Body'
    shape: 'Circle'
    angle: float
    position: tuple[float, float]
    space: Optional['Space'] = None


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
            config: 'CellConfig',
            group_id: int,
            position: pymunk.Vec2d,
            angle: float = 0.0,
            space: 'Space | None' = None
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
        self.shape.friction = 0. #TODO make it configurable
        self.shape.filter = pymunk.ShapeFilter(group=self.group_id)
        self.angle = angle
        self.position = position
        self.space = space

    @property
    def position(self) -> pymunk.Vec2d:
        """Returns the current position of the segment."""
        return self.body.position  # To return a tuple instead of Vec2d

    @position.setter
    def position(self, position: pymunk.Vec2d) -> None:
        assert isinstance(position, pymunk.Vec2d), "Position must be a pymunk.Vec2d"
        self.body.position = position

    @property
    def angle(self) -> float:
        """Returns the current angle of the segment."""
        return self.body.angle

    @angle.setter
    def angle(self, angle: float) -> None:
        self.body.angle = angle

    @property
    def radius(self) -> float:
        return self.shape.radius

    @radius.setter
    def radius(self, new_radius: float) -> None:
        friction, filter = self.shape.friction, self.shape.filter
        self.space.remove(self.shape)
        self.shape = pymunk.Circle(self.body, new_radius)
        self.shape.friction, self.shape.filter = friction, filter
        self.space.add(self.shape)
