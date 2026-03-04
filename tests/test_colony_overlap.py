from SyMBac.physics.colony import Colony


class _QueryInfo:
    def __init__(self, shape):
        self.shape = shape


class _Segment:
    def __init__(self, shape):
        self.shape = shape


class _PhysicsRepresentation:
    def __init__(self, shapes, tail_return=None, head_return=None):
        self.segments = [_Segment(shape) for shape in shapes]
        self._tail_return = tail_return
        self._head_return = head_return
        self.tail_calls = 0
        self.head_calls = 0

    def remove_tail_segment(self):
        self.tail_calls += 1
        return self._tail_return

    def remove_head_segment(self):
        self.head_calls += 1
        return self._head_return


class _Cell:
    def __init__(self, physics_representation):
        self.physics_representation = physics_representation


class _Space:
    def __init__(self, overlap_shape):
        self._overlap_shape = overlap_shape

    def shape_query(self, _shape):
        return [_QueryInfo(self._overlap_shape)]


def test_handle_cell_overlaps_stops_cleanly_when_removal_returns_none():
    overlap_shape = object()
    mother_pr = _PhysicsRepresentation(shapes=[overlap_shape], tail_return=None)
    daughter_pr = _PhysicsRepresentation(shapes=[object()], head_return=None)

    colony = Colony(space=_Space(overlap_shape), cells=[])
    colony.handle_cell_overlaps({_Cell(daughter_pr): _Cell(mother_pr)})

    assert mother_pr.tail_calls == 1
    assert daughter_pr.head_calls == 1
