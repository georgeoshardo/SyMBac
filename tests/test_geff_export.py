import json
import sys
import types

import numpy as np
from tifffile import imread

from SyMBac.lineage import Lineage
from SyMBac.renderer import Renderer


class _SimulationStub:
    def __init__(self, cell_timeseries, pix_mic_conv=0.1, resize_amount=2):
        self.cell_timeseries = cell_timeseries
        self.pix_mic_conv = pix_mic_conv
        self.resize_amount = resize_amount


class _CellStub:
    def __init__(
        self,
        mask_label,
        t,
        mother_mask_label=None,
        position=(0.0, 0.0),
        length=3.0,
        width=1.0,
        angle=0.0,
        generation=0,
    ):
        self.mask_label = mask_label
        self.t = t
        self.mother_mask_label = mother_mask_label
        self.position = position
        self.length = length
        self.width = width
        self.angle = angle
        self.generation = generation


def test_lineage_to_geff_exports_filtered_temporal_graph(monkeypatch):
    sim = _SimulationStub(
        [
            [_CellStub(mask_label=1, t=0, mother_mask_label=None, position=(10, 20))],
            [
                _CellStub(mask_label=1, t=1, mother_mask_label=None, position=(11, 21)),
                _CellStub(mask_label=2, t=1, mother_mask_label=1, position=(12, 22), generation=1),
            ],
        ]
    )
    lineage = Lineage(sim)

    captured = {}

    def _fake_write(graph, store, **kwargs):
        captured["graph"] = graph
        captured["store"] = store
        captured["kwargs"] = kwargs

    monkeypatch.setitem(sys.modules, "geff", types.SimpleNamespace(write=_fake_write))

    lineage.to_geff("dummy_store", frame_range=(1, 2), overwrite=True)

    graph = captured["graph"]
    kwargs = captured["kwargs"]

    assert captured["store"] == "dummy_store"
    assert kwargs["axis_names"] == ["t", "y", "x"]
    assert kwargs["axis_types"] == ["time", "space", "space"]
    assert kwargs["track_node_props"] == {"tracklet": "tracklet_id", "lineage": "lineage_id"}

    node_data = list(graph.nodes(data=True))
    assert len(node_data) == 2
    seg_ids = {attrs["seg_id"] for _, attrs in node_data}
    assert seg_ids == {1, 2}
    assert {attrs["t"] for _, attrs in node_data} == {1}
    assert len(graph.edges()) == 1


def test_generate_timeseries_training_data_writes_uint16_tiff_masks(tmp_path):
    renderer = Renderer.__new__(Renderer)
    renderer.additional_real_images = None
    renderer.params = types.SimpleNamespace(
        kwargs={
            "media_multiplier": 10.0,
            "cell_multiplier": 2.0,
            "device_multiplier": 5.0,
            "sigma": 1.5,
            "match_histogram": True,
            "match_noise": False,
            "match_fourier": False,
            "noise_var": 0.001,
            "defocus": 0.0,
            "halo_top_intensity": 1.0,
            "halo_bottom_intensity": 1.0,
            "halo_start": 0.0,
            "halo_end": 1.0,
            "cell_texture_strength": 0.0,
            "cell_texture_scale": 70.0,
            "edge_floor_opl": 0.0,
        }
    )
    renderer.simulation = types.SimpleNamespace(sim_length=5)

    def _fake_generate_test_comparison(**_kwargs):
        image = np.array([[0.0, 1.0], [0.5, 0.2]], dtype=np.float32)
        mask = np.array([[0, 300], [1, 2]], dtype=np.int32)
        superres = np.zeros_like(mask)
        return image, mask, superres

    renderer.generate_test_comparison = _fake_generate_test_comparison

    out_dir = tmp_path / "tracking_data"
    metadata = renderer.generate_timeseries_training_data(
        save_dir=str(out_dir),
        burn_in=1,
        sample_amount=0.0,
        n_series=2,
        frames_per_series=2,
        export_geff=False,
        seed=7,
        image_format="tiff",
    )

    assert metadata["n_series"] == 2
    assert metadata["simulation_frame_start"] == 1
    assert metadata["simulation_frame_end_exclusive"] == 3
    assert (out_dir / "metadata.json").exists()

    with open(out_dir / "series_000" / "manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert [frame["simulation_frame_idx"] for frame in manifest["frames"]] == [1, 2]

    mask_0 = imread(out_dir / "series_000" / "masks" / "frame_00000.tiff")
    assert mask_0.dtype == np.uint16
    assert int(mask_0.max()) == 300


def test_lineage_to_geff_falls_back_when_track_node_props_unsupported(monkeypatch):
    sim = _SimulationStub([[ _CellStub(mask_label=1, t=0, mother_mask_label=None, position=(1, 2)) ]])
    lineage = Lineage(sim)

    calls = {"count": 0}

    def _fake_write(_graph, _store, **kwargs):
        calls["count"] += 1
        if "track_node_props" in kwargs:
            raise TypeError("NxBackend.write() got an unexpected keyword argument 'track_node_props'")
        return None

    monkeypatch.setitem(sys.modules, "geff", types.SimpleNamespace(write=_fake_write))
    lineage.to_geff("dummy_store_fallback", overwrite=True)
    assert calls["count"] == 2
