import json
import sys
import types

import geff
import numpy as np
import pytest
from geff.validate.data import ValidationConfig
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


def _timeseries_renderer(mask_label):
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
        mask = np.array([[0, mask_label], [1, 2]], dtype=np.int64)
        superres = np.zeros_like(mask)
        return image, mask, superres

    renderer.generate_test_comparison = _fake_generate_test_comparison
    return renderer


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

    monkeypatch.setitem(
        sys.modules,
        "geff",
        types.SimpleNamespace(write=_fake_write, GeffMetadata=geff.GeffMetadata),
    )

    lineage.to_geff("dummy_store", frame_range=(1, 2), overwrite=True)

    graph = captured["graph"]
    kwargs = captured["kwargs"]

    assert captured["store"] == "dummy_store"
    assert kwargs["axis_names"] == ["t", "y", "x"]
    assert kwargs["axis_types"] == ["time", "space", "space"]
    assert kwargs["axis_units"] == ["frame", "micrometer", "micrometer"]
    assert kwargs["metadata"].track_node_props == {
        "tracklet": "tracklet_id",
        "lineage": "lineage_id",
    }

    node_data = list(graph.nodes(data=True))
    assert len(node_data) == 2
    seg_ids = {attrs["seg_id"] for _, attrs in node_data}
    assert seg_ids == {1, 2}
    assert {attrs["t"] for _, attrs in node_data} == {1}
    assert len(graph.edges()) == 1
    daughter = next(attrs for _, attrs in node_data if attrs["seg_id"] == 2)
    assert daughter["x"] == pytest.approx(0.6)
    assert daughter["y"] == pytest.approx(1.1)
    assert daughter["length"] == pytest.approx(0.15)
    assert daughter["width"] == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("mask_label", "expected_dtype"),
    [(300, np.uint16), (65536, np.uint32)],
)
def test_generate_timeseries_training_data_promotes_inferred_mask_dtype(
    tmp_path, mask_label, expected_dtype
):
    renderer = _timeseries_renderer(mask_label)

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
    assert mask_0.dtype == expected_dtype
    assert int(mask_0.max()) == mask_label
    assert metadata["mask_dtype"] == "inferred"
    assert manifest["frames"][0]["mask_dtype"] == np.dtype(expected_dtype).name


def test_generate_timeseries_training_data_rejects_insufficient_explicit_dtype(tmp_path):
    renderer = _timeseries_renderer(256)

    with pytest.raises(ValueError, match="label range.*uint8"):
        renderer.generate_timeseries_training_data(
            save_dir=str(tmp_path / "tracking_data"),
            burn_in=1,
            sample_amount=0.0,
            n_series=1,
            frames_per_series=1,
            mask_dtype=np.uint8,
            export_geff=False,
            seed=7,
            image_format="tiff",
        )


def test_generate_timeseries_training_data_rejects_uint32_png_masks(tmp_path):
    renderer = _timeseries_renderer(65536)

    with pytest.raises(ValueError, match="PNG.*uint16.*TIFF"):
        renderer.generate_timeseries_training_data(
            save_dir=str(tmp_path / "tracking_data"),
            burn_in=1,
            sample_amount=0.0,
            n_series=1,
            frames_per_series=1,
            export_geff=False,
            seed=7,
            image_format="png",
        )


def test_lineage_to_geff_round_trips_standard_metadata(tmp_path):
    sim = _SimulationStub(
        [
            [_CellStub(mask_label=1, t=0, position=(1, 2))],
            [
                _CellStub(mask_label=1, t=1, position=(1, 2)),
                _CellStub(mask_label=2, t=1, mother_mask_label=1, position=(1, 2)),
            ],
            [
                _CellStub(mask_label=1, t=2, position=(1, 2)),
                _CellStub(mask_label=2, t=2, mother_mask_label=1, position=(1, 2)),
            ],
        ]
    )
    lineage = Lineage(sim)

    store = tmp_path / "lineage.geff"
    lineage.to_geff(store, frame_range=(2, 3), overwrite=True)
    _, metadata = geff.read(
        store,
        data_validation=ValidationConfig(graph=True, tracklet=True, lineage=True),
    )

    assert metadata.track_node_props == {
        "tracklet": "tracklet_id",
        "lineage": "lineage_id",
    }
    assert [axis.unit for axis in metadata.axes] == [
        "frame",
        "micrometer",
        "micrometer",
    ]
    assert metadata.node_props_metadata["x"].unit == "micrometer"
    assert metadata.node_props_metadata["y"].unit == "micrometer"
    assert metadata.node_props_metadata["length"].unit == "micrometer"
    assert metadata.node_props_metadata["width"].unit == "micrometer"


def test_lineage_to_geff_uses_pixel_units_without_spatial_calibration(tmp_path):
    sim = _SimulationStub(
        [[_CellStub(mask_label=1, t=0, position=(10, 20), length=3, width=1)]],
        pix_mic_conv=None,
    )
    store = tmp_path / "lineage.geff"

    Lineage(sim).to_geff(store, overwrite=True)
    graph, metadata = geff.read(store)

    node = next(iter(graph.nodes.values()))
    assert node["x"] == 10
    assert node["y"] == 20
    assert node["length"] == 3
    assert node["width"] == 1
    assert [axis.unit for axis in metadata.axes] == ["frame", "pixel", "pixel"]
    assert metadata.node_props_metadata["length"].unit == "pixel"
    assert metadata.node_props_metadata["width"].unit == "pixel"
