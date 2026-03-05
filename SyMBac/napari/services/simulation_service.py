from __future__ import annotations

from SyMBac.config_models import SimulationSpec
from SyMBac.simulation import Simulation


class SimulationService:
    def build(self, spec: SimulationSpec) -> Simulation:
        return Simulation(spec)

    def run(self, simulation: Simulation, show_window: bool = False) -> Simulation:
        simulation.run(show_window=show_window)
        return simulation

    def draw_opl(self, simulation: Simulation, do_transformation: bool = False, label_masks: bool = True) -> Simulation:
        simulation.draw_opl(do_transformation=do_transformation, label_masks=label_masks)
        return simulation
