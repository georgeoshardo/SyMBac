"""Cells leave the trench only once more than ``exit_fraction`` of them is outside.

Previously any single segment past the open end removed the whole cell, so the bottom
of a synthetic trench was always empty and a full trench could not be simulated,
unlike cropped real images which routinely show a partial cell at the mouth.
"""
import numpy as np
import pytest

from SyMBac.physics.microfluidic_geometry import GeometryLayout, TrenchGeometrySpec


def _cell_with_fraction_outside(spec, layout, fraction, n=10, radius=4.0):
    """A straight vertical cell of n segments whose leading `fraction` lies past the open end."""
    n_out = int(round(fraction * n))
    spacing = 2.0
    # Segment centres straddle the open end: n_out of them clearly beyond, the rest clearly inside.
    ys = np.array([spec.open_end_y + spacing * (i + 1) for i in range(n_out)]
                  + [spec.open_end_y - spacing * (i + 1) for i in range(n - n_out)])
    local = np.column_stack([np.zeros(n), ys])
    return layout.to_world_points(local), np.full(n, radius)


@pytest.mark.parametrize("fraction, removed", [(0.0, False), (0.3, False), (0.5, False), (0.6, True), (1.0, True)])
def test_cell_removed_only_when_more_than_half_has_left(fraction, removed):
    spec = TrenchGeometrySpec(width=20.0, trench_length=60.0)
    layout = GeometryLayout(spec)
    positions, radii = _cell_with_fraction_outside(spec, layout, fraction)
    assert spec.cell_out_of_bounds(positions, radii, layout) is removed


def test_exit_fraction_zero_restores_any_segment_rule():
    spec = TrenchGeometrySpec(width=20.0, trench_length=60.0, exit_fraction=0.0)
    layout = GeometryLayout(spec)
    positions, radii = _cell_with_fraction_outside(spec, layout, 0.1)
    assert spec.cell_out_of_bounds(positions, radii, layout)


def test_segment_above_closed_end_is_always_culled():
    spec = TrenchGeometrySpec(width=20.0, trench_length=60.0)
    layout = GeometryLayout(spec)
    positions = layout.to_world_points(np.array([[0.0, -3.0], [0.0, 5.0], [0.0, 9.0]]))
    assert spec.cell_out_of_bounds(positions, np.full(3, 4.0), layout)


def test_jitter_trials_allow_protrusion_but_not_culling():
    spec = TrenchGeometrySpec(width=20.0, trench_length=60.0)
    layout = GeometryLayout(spec)
    protruding, radii = _cell_with_fraction_outside(spec, layout, 0.3)
    assert spec.positions_within_bounds(protruding, radii, layout, enforce_open_end_cap=True)
    leaving, radii = _cell_with_fraction_outside(spec, layout, 0.7)
    assert not spec.positions_within_bounds(leaving, radii, layout, enforce_open_end_cap=True)
    assert spec.positions_within_bounds(leaving, radii, layout, enforce_open_end_cap=False)


def test_exit_fraction_validation():
    with pytest.raises(ValueError):
        TrenchGeometrySpec(width=20.0, trench_length=60.0, exit_fraction=1.0)
