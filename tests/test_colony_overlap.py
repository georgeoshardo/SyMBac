from types import SimpleNamespace

import pytest

from SyMBac.physics.colony import Colony


class _QueryInfo:
    def __init__(self, shape):
        self.shape = shape


class _Segment:
    def __init__(self, shape):
        self.shape = shape


class _PhysicsRepresentation:
    def __init__(self, shapes, minimum_segments):
        self.segments = [_Segment(shape) for shape in shapes]
        self.minimum_segments = minimum_segments
        self.tail_calls = 0
        self.head_calls = 0

    def remove_tail_segment(self):
        self.tail_calls += 1
        if len(self.segments) <= self.minimum_segments:
            return None
        return self.segments.pop()

    def remove_head_segment(self):
        self.head_calls += 1
        if len(self.segments) <= self.minimum_segments:
            return None
        return self.segments.pop(0)


class _Cell:
    def __init__(self, physics_representation, minimum_segments):
        self.physics_representation = physics_representation
        self.config = SimpleNamespace(MIN_LENGTH_AFTER_DIVISION=minimum_segments)


class _Space:
    def __init__(self, overlap_shape):
        self._overlap_shape = overlap_shape

    def shape_query(self, _shape):
        return [_QueryInfo(self._overlap_shape)]


@pytest.mark.parametrize("mother_segments,daughter_segments", [(2, 3), (3, 2)])
def test_handle_cell_overlaps_validates_both_cells_before_mutating(
    mother_segments,
    daughter_segments,
):
    minimum_segments = 2
    overlap_shape = object()
    mother_pr = _PhysicsRepresentation(
        shapes=[overlap_shape, *[object() for _ in range(mother_segments - 1)]],
        minimum_segments=minimum_segments,
    )
    daughter_pr = _PhysicsRepresentation(
        shapes=[object() for _ in range(daughter_segments)],
        minimum_segments=minimum_segments,
    )
    mother = _Cell(mother_pr, minimum_segments)
    daughter = _Cell(daughter_pr, minimum_segments)

    colony = Colony(space=_Space(overlap_shape), cells=[])
    colony.handle_cell_overlaps({daughter: mother})

    assert mother_pr.tail_calls == 0
    assert daughter_pr.head_calls == 0
    assert len(mother_pr.segments) == mother_segments
    assert len(daughter_pr.segments) == daughter_segments
