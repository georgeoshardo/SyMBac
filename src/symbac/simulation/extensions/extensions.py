import typing

from symbac.trench_geometry import box_creator

if typing.TYPE_CHECKING:
    from symbac.simulation import Simulator


class SimulationExtensions:

    def __init__(self, simulator: Simulator) -> None:
        self.simulator = simulator
        self.extensions = []

    def add_box(self):
        box_creator(1000, 1000, (0, 0), self.simulator.space, barrier_thickness=10, fillet_radius=100)
