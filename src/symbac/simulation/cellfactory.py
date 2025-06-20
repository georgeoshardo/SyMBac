import pymunk.pygame_util
from symbac.simulation.config import CellConfig


class CellFactory:
    def __init__(self, space: pymunk.Space, config: CellConfig) -> None:
        self.space = space
        self.config = config

