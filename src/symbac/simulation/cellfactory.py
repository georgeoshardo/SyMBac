import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np
from typing import Optional
from symbac.misc import generate_color
from symbac.simulation.config import CellConfig
from symbac.simulation.cell import Cell

class CellFactory:
    def __init__(self, space: pymunk.Space, config: CellConfig) -> None:
        self.space = space
        self.config = config

    def create_seed_cell(self, start_pos: tuple[float, float]) -> Cell:

        bodies: list[pymunk.Body] = []
        shapes: list[pymunk.Circle] = []

        pivot_joints: list[pymunk.PivotJoint] = []
        limit_joints: list[pymunk.RotaryLimitJoint] = []
        spring_joints: list[pymunk.DampedRotarySpring] = []


        for i in range(self.config.SEED_CELL_SEGMENTS):
            self._add_seed_cell_segments(i == 0)


    def _add_seed_cell_segments(self, is_first: bool) -> None:
        """Adds and joins segments during the initial construction of the very first cell.

        This method creates a pymunk body and shape for one segment. It
        has two modes:
        1. If `is_first` is True, it places the segment at the cell's
           designated start position.
        2. If `is_first` is False, it places the segment adjacent to the
           previously added segment and creates the necessary physical joints
           (Pivot, and optionally Rotary Limit and Spring) to connect them.

        Args:
            is_first: A flag to indicate if this is the very first segment
                      of the cell.
        """
        moment = pymunk.moment_for_circle(
            self.config.SEGMENT_MASS, 0, self.config.SEGMENT_RADIUS
        )
        body = pymunk.Body(self.config.SEGMENT_MASS, moment)

        shape = pymunk.Circle(body, self.config.SEGMENT_RADIUS)
        shape.friction = 0.0
        shape.filter = pymunk.ShapeFilter(group=self.group_id)

        if is_first:
            body.position = self.start_pos
        else:
            print(self.group_id)
            prev_body = self.bodies[-1]
            offset = Vec2d(self.joint_distance, 0).rotated(prev_body.angle)
            body.position = prev_body.position + offset
            body.angle = prev_body.angle
            noise_x = np.random.uniform(-0.1, 0.1)
            noise_y = np.random.uniform(-0.1, 0.1)
            body.position += Vec2d(noise_x, noise_y)



        self.space.add(body, shape)
        self.bodies.append(body)
        self.shapes.append(shape)

        if not is_first:
            prev_body = self.bodies[-2]

            anchor_on_prev = (self.joint_distance / 2, 0)
            anchor_on_curr = (-self.joint_distance / 2, 0)
            pivot = pymunk.PivotJoint(
                prev_body, body, anchor_on_prev, anchor_on_curr
            )
            pivot.max_force = self.config.PIVOT_JOINT_STIFFNESS # - pivots should have a hardcoded max force I think?
            self.space.add(pivot)

            if self.config.ROTARY_LIMIT_JOINT:
                assert self.config.STIFFNESS is not None
                assert self.config.MAX_BEND_ANGLE is not None
                limit = pymunk.RotaryLimitJoint(
                    prev_body, body, -self.config.MAX_BEND_ANGLE, self.config.MAX_BEND_ANGLE
                )
                limit.max_force = self.config.STIFFNESS
                self.space.add(limit)
                self.limit_joints.append(limit)

            if self.config.DAMPED_ROTARY_SPRING:
                assert self.config.ROTARY_SPRING_STIFFNESS is not None
                assert self.config.ROTARY_SPRING_DAMPING is not None
                spring = pymunk.DampedRotarySpring(
                    prev_body, body, 0, self.config.ROTARY_SPRING_STIFFNESS, self.config.ROTARY_SPRING_DAMPING
                )
                self.space.add(spring)
                self.spring_joints.append(spring)

            self.pivot_joints.append(pivot)