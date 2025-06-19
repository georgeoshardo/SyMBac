import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np
from typing import Optional
from symbac.misc import generate_color
from symbac.simulation.config import CellConfig


class CellSegment:
    def __init__(self, config: CellConfig, group_id: int) -> None:
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