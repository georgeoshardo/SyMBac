from copy import deepcopy

from joblib import Parallel, delayed
from tqdm import tqdm

from symbac.simulation import Simulation


class ParallelSimulation:

    def __init__(self, simulation, N):
        self.simulations = [deepcopy(simulation) for x in range(N)]

        for i, simulation in enumerate(self.simulations):
            init_args = simulation.__dict__
            init_args["save_dir"] += f"/parsim_{str(i)}/"
            simulation.__init__(**init_args)
            simulation.show_progress = False

    def run_simulations(self):
        def run_single_simulation(simulation):  # Needed to avoid joblib's shared mem requirement
            simulation.run_simulation()
            return simulation

        self.simulations = Parallel(n_jobs=-1)(delayed(run_single_simulation)(sim) for sim in tqdm(self.simulations))
