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

