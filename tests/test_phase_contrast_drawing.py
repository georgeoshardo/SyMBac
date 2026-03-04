import pymunk

from SyMBac.cell_simulation import run_simulation
from SyMBac.drawing import draw_scene, generate_curve_props, gen_cell_props_for_draw, get_space_size
from SyMBac.trench_geometry import get_trench_segments


def test_phase_contrast_drawing_pipeline_smoke(tmp_path):
    cell_timeseries, space, historic_cells = run_simulation(
        trench_length=15,
        trench_width=1.5,
        cell_max_length=6,
        cell_width=1,
        sim_length=3,
        pix_mic_conv=0.0655,
        gravity=0,
        phys_iters=2,
        max_length_std=0.3,
        width_std=0.1,
        save_dir=str(tmp_path),
        resize_amount=1,
        lysis_p=0.0,
        show_window=False,
    )

    assert isinstance(space, pymunk.Space)
    assert len(historic_cells) > 0

    main_segments = get_trench_segments(space)
    assert len(main_segments) == 2

    id_props = generate_curve_props(cell_timeseries)
    cell_timeseries_properties = [
        gen_cell_props_for_draw(frame, id_props)
        for frame in cell_timeseries
    ]
    assert len(cell_timeseries_properties) > 0

    scene, masks = draw_scene(
        cell_timeseries_properties[0],
        do_transformation=True,
        space_size=get_space_size(cell_timeseries_properties),
        offset=30,
        label_masks=True,
    )

    assert scene.shape == masks.shape
    assert scene.ndim == 2
