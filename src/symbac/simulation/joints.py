import pymunk
import pymunk.pygame_util
from symbac.simulation.config import CellConfig

class CellJoint(pymunk.PivotJoint):
    """
    A custom PivotJoint that uses the configuration from CellConfig.
    """
    def __init__(self, body_a: pymunk.Body, body_b: pymunk.Body, config: CellConfig) -> None:
        self.joint_distance = config.SEGMENT_RADIUS / config.GRANULARITY
        anchor_on_prev = (self.joint_distance / 2, 0)  # Local coordinates relative to the body for the pivot joint
        anchor_on_curr = (-self.joint_distance / 2, 0)

        super().__init__(body_a, body_b, anchor_on_prev, anchor_on_curr)
        self.max_force = config.PIVOT_JOINT_STIFFNESS


