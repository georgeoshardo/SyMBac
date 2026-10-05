import sys

import pytest


def test_colony_modules_import_without_optional_dependencies():
    from SyMBac.colony_renderer import ColonyRenderer
    from SyMBac.colony_simulation import ColonySimulation

    assert ColonyRenderer.__name__ == "ColonyRenderer"
    assert ColonySimulation.__name__ == "ColonySimulation"


def test_cellmodeller_is_required_only_when_simulation_is_run(monkeypatch):
    from SyMBac.colony_simulation import ColonySimulation

    monkeypatch.setitem(sys.modules, "CellModeller", None)
    simulation = ColonySimulation.__new__(ColonySimulation)

    with pytest.raises(ImportError, match="CellModeller.*run_cellmodeller_sim"):
        simulation.run_cellmodeller_sim(1)


def test_ray_is_required_only_for_ray_generation(monkeypatch):
    from SyMBac.colony_renderer import ColonyRenderer

    monkeypatch.setitem(sys.modules, "ray", None)
    renderer = ColonyRenderer.__new__(ColonyRenderer)

    with pytest.raises(ImportError, match="Ray.*generate_random_samples_ray"):
        renderer.generate_random_samples_ray(1, 0, "unused")
