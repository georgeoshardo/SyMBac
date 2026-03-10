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
