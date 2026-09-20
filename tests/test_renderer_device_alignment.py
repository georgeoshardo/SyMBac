"""The synthetic device must be drawn in the same frame as the simulated cells.

Regression test for the closed-end gap: the renderer used to rebuild the trench from
pymunk bounding boxes with the closed end hard-coded at world y = 0, while the physics
placed the closed end at ``GeometryLayout.world_offset[1]``. Cells touching the real
closed end therefore appeared ~world_offset below the drawn cap.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from SyMBac.drawing import draw_scene_from_segments
from SyMBac.physics.microfluidic_geometry import GeometryLayout, TrenchGeometrySpec
from SyMBac.renderer import Renderer

OFFSET = 30


def _trench(width=20.0, length=60.0):
    spec = TrenchGeometrySpec(width=width, trench_length=length, barrier_thickness=10.0)
    return spec, GeometryLayout(spec)


def test_interior_mask_apex_is_at_local_origin_and_matches_bounds_test():
    spec, layout = _trench()
    shape = (int(layout.preview_shape[0]) + OFFSET, int(layout.preview_shape[1]) + OFFSET)
    interior = spec.interior_mask(layout, shape, OFFSET)

    apex_row = int(round(layout.world_offset[1] + OFFSET))
    centre_col = int(round(layout.world_offset[0] + OFFSET))
    rows_in_centre_column = np.flatnonzero(interior[:, centre_col])
    assert rows_in_centre_column.min() == apex_row
    # Pixels are interior while local y <= open_end_y, so the last interior row is the floor.
    assert rows_in_centre_column.max() == int(np.floor(layout.world_offset[1] + spec.open_end_y + OFFSET))

    # The raster agrees with the geometry's own point-in-bounds test on sampled points.
    rng = np.random.default_rng(0)
    local = rng.uniform([-spec.inner_half_width - 5, -5], [spec.inner_half_width + 5, spec.open_end_y + 5], size=(400, 2))
    for lx, ly in local:
        wx, wy = layout.to_world_point((lx, ly))
        row, col = int(round(wy + OFFSET)), int(round(wx + OFFSET))
        if not (0 <= row < shape[0] and 0 <= col < shape[1]):
            continue
        inside_geometry = spec.positions_within_bounds(
            [(wx, wy)], [0.0], layout, enforce_open_end_cap=True
        ) and ly >= spec.inner_half_width - np.sqrt(max(spec.inner_half_width**2 - lx**2, 0.0))
        if abs(abs(lx) - spec.inner_half_width) < 1.0 or abs(ly) < 1.0 or abs(ly - spec.open_end_y) < 1.0:
            continue  # skip pixels straddling an edge
        assert bool(interior[row, col]) == bool(inside_geometry), (lx, ly)


def test_scene_window_starts_at_closed_end_and_spans_walls():
    spec, layout = _trench()
    row_start, row_stop, col_start, col_stop = spec.scene_window(layout, OFFSET)
    assert row_start == int(round(layout.world_offset[1] + OFFSET))
    assert row_stop == int(round(layout.world_bounds.max_y + OFFSET))
    assert col_start == int(round(layout.world_bounds.min_x + OFFSET))
    assert col_stop == int(round(layout.world_bounds.max_x + OFFSET))
    assert (col_stop - col_start) == pytest.approx(layout.local_bounds.width, abs=1)


def _renderer_for(spec, layout):
    renderer = Renderer.__new__(Renderer)
    simulation = SimpleNamespace(offset=OFFSET)
    simulation.ensure_geometry_layout = lambda: (spec, layout)
    renderer.simulation = simulation
    renderer.PSF = SimpleNamespace(mode="phase contrast")
    return renderer


def test_rendered_cell_touching_closed_end_touches_drawn_cap():
    spec, layout = _trench()
    radius = 4.0
    # One cell resting against the closed end, exactly as the physics places a seed cell.
    local_positions = np.array([[0.0, radius + i * 1.5] for i in range(6)])
    world_positions = layout.to_world_points(local_positions)
    cells = [{"positions": world_positions, "radii": np.full(len(world_positions), radius),
              "mask_label": 1, "cell_id": 1}]
    shape = (int(layout.preview_shape[0]) + OFFSET, int(layout.preview_shape[1]) + OFFSET)
    scene, mask = draw_scene_from_segments(cells, shape, OFFSET, label_masks=True)

    renderer = _renderer_for(spec, layout)
    media, cell, device = 75.0, 1.7, 29.0
    expanded_scene, expanded_no_cells, expanded_mask = renderer.generate_PC_OPL(
        scene, mask, media, cell, device,
        y_border_expansion_coefficient=2, x_border_expansion_coefficient=2, defocus=0.0,
    )

    interior = expanded_no_cells == device
    centre_col = expanded_no_cells.shape[1] // 2
    cap_apex_row = np.flatnonzero(interior[:, centre_col]).min()
    cell_top_row = np.flatnonzero(expanded_mask[:, centre_col]).min()
    # The cell's top is at local y = 0 (radius above its first centre) -> same row as the apex.
    assert 0 <= cell_top_row - cap_apex_row <= 1

    # The interior is the semicircle-capped channel: nothing of the cell is masked away.
    assert expanded_mask.sum() == pytest.approx(mask.sum(), rel=0.02)
    # And nothing is drawn as interior above the apex.
    assert not interior[:cap_apex_row].any()


def test_fluorescence_mode_keeps_whole_scene_and_mask():
    spec, layout = _trench()
    shape = (int(layout.preview_shape[0]) + OFFSET, int(layout.preview_shape[1]) + OFFSET)
    scene = np.zeros(shape)
    mask = np.zeros(shape, dtype=int)
    r, c = int(round(layout.world_offset[1] + OFFSET)) + 10, int(round(layout.world_offset[0] + OFFSET))
    scene[r, c] = 3.0
    mask[r, c] = 7
    renderer = _renderer_for(spec, layout)
    renderer.PSF = SimpleNamespace(mode="fluorescence")
    expanded_scene, _, expanded_mask = renderer.generate_PC_OPL(
        scene, mask, 75.0, 1.0, 29.0, y_border_expansion_coefficient=2, x_border_expansion_coefficient=2, defocus=0.0
    )
    assert expanded_scene.max() == pytest.approx(3.0)
    assert expanded_mask.max() == 7
