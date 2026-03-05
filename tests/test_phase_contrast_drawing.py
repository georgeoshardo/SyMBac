import pymunk

from SyMBac.config_models import (
    SimulationCellSpec,
    SimulationGeometrySpec,
    SimulationPhysicsSpec,
    SimulationRuntimeSpec,
    SimulationSpec,
)
from SyMBac.simulation import Simulation
from SyMBac.trench_geometry import get_trench_segments


def _simulation_spec(tmp_path):
    return SimulationSpec(
        geometry=SimulationGeometrySpec(
            trench_length=15.0,
            trench_width=1.5,
            pix_mic_conv=0.0655,
            resize_amount=1,
        ),
        cell=SimulationCellSpec(
            cell_max_length=6.0,
            cell_width=1.0,
            max_length_std=0.3,
            width_std=0.1,
            lysis_p=0.0,
        ),
        physics=SimulationPhysicsSpec(gravity=0.0, phys_iters=2),
        runtime=SimulationRuntimeSpec(
            sim_length=2,
            substeps=1,
            save_dir=str(tmp_path / "sim"),
        ),
    )


def test_phase_contrast_drawing_pipeline_smoke(tmp_path):
    simulation = Simulation(_simulation_spec(tmp_path))
    simulation.run(show_window=False)

    assert isinstance(simulation.space, pymunk.Space)
    assert len(simulation.cell_timeseries) > 0

    main_segments = get_trench_segments(simulation.space)
    assert len(main_segments) == 2

    opl_scenes, masks = simulation.draw_opl(
        do_transformation=True,
        label_masks=True,
        return_output=True,
    )

    assert len(opl_scenes) == simulation.sim_length
    assert len(masks) == simulation.sim_length
    assert opl_scenes[0].shape == masks[0].shape
    assert opl_scenes[0].ndim == 2
