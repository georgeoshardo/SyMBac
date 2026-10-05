import pytest

from SyMBac.lineage import Lineage


class SimulationStub:
    def __init__(self, cell_timeseries):
        self.cell_timeseries = cell_timeseries


class SlotOnlyCell:
    __slots__ = ("mask_label", "mother_mask_label", "t")

    def __init__(self, mask_label, t, mother_mask_label=None):
        self.mask_label = mask_label
        self.mother_mask_label = mother_mask_label
        self.t = t


class DivisionAwareCell(SlotOnlyCell):
    __slots__ = ("just_divided",)

    def __init__(self, mask_label, t, mother_mask_label=None, just_divided=False):
        super().__init__(mask_label, t, mother_mask_label)
        self.just_divided = just_divided


def _edge_set(edgelist):
    return {tuple(edge) for edge in edgelist.tolist()}


def test_lineage_handles_slot_only_cells():
    sim = SimulationStub(
        [
            [SlotOnlyCell(mask_label=1, t=0)],
            [
                SlotOnlyCell(mask_label=1, t=1),
                SlotOnlyCell(mask_label=2, t=1, mother_mask_label=1),
            ],
        ]
    )

    lineage = Lineage(sim)

    assert (1, 2) in _edge_set(lineage.family_tree_edgelist)
    assert lineage.temporal_lineage_graph.has_edge((1, 0), (1, 1))
    assert lineage.temporal_lineage_graph.has_edge((1, 1), (2, 1))


def test_lineage_adds_persistent_mother_metadata_edge_only_at_division_frame():
    sim = SimulationStub(
        [
            [DivisionAwareCell(mask_label=1, t=0)],
            [
                DivisionAwareCell(mask_label=1, t=1, just_divided=True),
                DivisionAwareCell(
                    mask_label=2,
                    t=1,
                    mother_mask_label=1,
                    just_divided=True,
                ),
            ],
            [
                DivisionAwareCell(mask_label=1, t=2),
                DivisionAwareCell(mask_label=2, t=2, mother_mask_label=1),
            ],
        ]
    )

    lineage = Lineage(sim)

    assert lineage.temporal_lineage_graph.has_edge((1, 1), (2, 1))
    assert not lineage.temporal_lineage_graph.has_edge((1, 2), (2, 2))


def test_lineage_does_not_treat_late_first_observation_as_division():
    sim = SimulationStub(
        [
            [
                DivisionAwareCell(mask_label=1, t=5),
                DivisionAwareCell(mask_label=2, t=5, mother_mask_label=1),
            ]
        ]
    )

    lineage = Lineage(sim)

    assert not lineage.temporal_lineage_graph.has_edge((1, 5), (2, 5))


def test_lineage_rejects_duplicate_mask_label_and_time():
    sim = SimulationStub(
        [[SlotOnlyCell(mask_label=1, t=0), SlotOnlyCell(mask_label=1, t=0)]]
    )

    with pytest.raises(ValueError, match="Duplicate detection.*mask_label=1.*time=0"):
        Lineage(sim)


def test_lineage_handles_mixed_frames_and_missing_mother_node():
    sim = SimulationStub(
        [
            [
                SlotOnlyCell(mask_label=1, t=0),
                SlotOnlyCell(mask_label=3, t=0, mother_mask_label=None),
            ],
            [
                SlotOnlyCell(mask_label=2, t=1, mother_mask_label=1),
                SlotOnlyCell(mask_label=3, t=1, mother_mask_label=None),
                SlotOnlyCell(mask_label=4, t=1, mother_mask_label=3),
            ],
        ]
    )

    with pytest.warns(RuntimeWarning, match="expected mother temporal node"):
        lineage = Lineage(sim)

    edge_set = _edge_set(lineage.family_tree_edgelist)
    assert (1, 2) in edge_set
    assert (3, 4) in edge_set
    assert not lineage.temporal_lineage_graph.has_edge((1, 1), (2, 1))
    assert lineage.temporal_lineage_graph.has_edge((3, 0), (3, 1))
    assert lineage.temporal_lineage_graph.has_edge((3, 1), (4, 1))
