import pymunk
from symbac.simulation.config import CellConfig
from symbac.simulation.segments import CellSegment

class CellJoint(pymunk.PivotJoint):
    """Represents a custom PivotJoint connecting two CellSegments.

    This joint uses parameters from a CellConfig object to define its
    physical properties, such as the distance between connected segments
    and its stiffness.

    Attributes:
        joint_distance: The calculated distance between the anchor
            points on the connected segments. This is derived from
            `config.SEGMENT_RADIUS` and `config.GRANULARITY`.
        max_force: The maximum force this pivot joint can exert,
            derived from `config.PIVOT_JOINT_STIFFNESS`.
    """
    def __init__(self, segment_a: CellSegment, segment_b: CellSegment, config: CellConfig) -> None:
        """Initializes a CellJoint instance.

        This constructor sets up a `pymunk.PivotJoint` between two
        `CellSegment` objects. It calculates the necessary `joint_distance`
        and sets the `max_force` based on the provided `CellConfig`.

        Args:
            segment_a: The first CellSegment to connect.
            segment_b: The second CellSegment to connect.
            config: The CellConfig object providing joint parameters.
        """
        self.joint_distance: float = config.SEGMENT_RADIUS / config.GRANULARITY
        anchor_on_prev = (self.joint_distance / 2, 0)  # Local coordinates relative to the body for the pivot joint
        anchor_on_curr = (-self.joint_distance / 2, 0)

        body_a = segment_a.body
        body_b = segment_b.body

        super().__init__(body_a, body_b, anchor_on_prev, anchor_on_curr)
        self.max_force: float = config.PIVOT_JOINT_STIFFNESS


class CellRotaryLimitJoint(pymunk.RotaryLimitJoint):
    """
    A custom RotaryLimitJoint that uses the configuration from CellConfig.
    """
    def __init__(self, segment_a: CellSegment, segment_b: CellSegment, config: CellConfig) -> None:
        assert config.STIFFNESS is not None
        assert config.MAX_BEND_ANGLE is not None
        assert config.ROTARY_LIMIT_JOINT

        body_a = segment_a.body
        body_b = segment_b.body

        super().__init__(body_a, body_b, -config.MAX_BEND_ANGLE, config.MAX_BEND_ANGLE)
        self.max_force = config.STIFFNESS

class CellDampedRotarySpring(pymunk.DampedRotarySpring):
    """
    A custom DampedRotarySpring that uses the configuration from CellConfig.
    """
    def __init__(self, segment_a: CellSegment, segment_b: CellSegment, config: CellConfig) -> None:
        assert config.DAMPED_ROTARY_SPRING
        assert config.ROTARY_SPRING_STIFFNESS is not None
        assert config.ROTARY_SPRING_DAMPING is not None

        body_a = segment_a.body
        body_b = segment_b.body

        super().__init__(body_a, body_b, 0.0, config.ROTARY_SPRING_STIFFNESS, config.ROTARY_SPRING_DAMPING)