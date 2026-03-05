import sys
import types

import numpy as np
import yaml
from tifffile import imread

from SyMBac.config_models import (
    DatasetOutputConfig,
    RenderConfig,
    RenderResult,
    TimeseriesDatasetPlan,
)
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


def test_export_dataset_writes_uint16_tiff_masks(tmp_path):
    renderer = Renderer.__new__(Renderer)
    renderer.additional_real_images = None
    renderer.simulation = types.SimpleNamespace(sim_length=5)
    renderer._ensure_image_params = lambda *args, **kwargs: None

    def _fake_render_frame(_scene_no, config, real_image_override=None):
        del config, real_image_override
        image = np.array([[0.0, 1.0], [0.5, 0.2]], dtype=np.float32)
        mask = np.array([[0, 300], [1, 2]], dtype=np.int32)
        return RenderResult(image=image, mask=mask, superres_mask=np.zeros_like(mask))

    renderer.render_frame = _fake_render_frame

    out_dir = tmp_path / "tracking_data"
    metadata = renderer.export_dataset(
        plan=TimeseriesDatasetPlan(
            burn_in=1,
            sample_amount=0.0,
            n_series=2,
            frames_per_series=2,
        ),
        output=DatasetOutputConfig(
            save_dir=str(out_dir),
            image_format="tiff",
            mask_dtype="uint16",
            export_geff=False,
        ),
        base_config=RenderConfig(),
        seed=7,
    )

    assert metadata["n_series"] == 2
    assert metadata["simulation_frame_start"] == 1
    assert metadata["simulation_frame_end_exclusive"] == 3
    assert (out_dir / "metadata.yaml").exists()

    with open(out_dir / "series_000" / "manifest.yaml", "r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    assert [frame["simulation_frame_idx"] for frame in manifest["frames"]] == [1, 2]

    mask_0 = imread(out_dir / "series_000" / "masks" / "frame_00000.tiff")
    assert mask_0.dtype == np.uint16
    assert int(mask_0.max()) == 300


def test_lineage_to_geff_falls_back_when_track_node_props_unsupported(monkeypatch):
    sim = _SimulationStub([[_CellStub(mask_label=1, t=0, mother_mask_label=None, position=(1, 2))]])
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
