from pathlib import Path

import yaml


def test_napari_manifest_has_five_docks():
    manifest_path = Path("SyMBac/napari/resources/napari.yaml")
    assert manifest_path.exists()

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    widgets = data["contributions"]["widgets"]
    display_names = {widget["display_name"] for widget in widgets}

    assert len(widgets) == 5
    assert display_names == {
        "SyMBac Simulation",
        "SyMBac Optics",
        "SyMBac Regions",
        "SyMBac Tuning",
        "SyMBac Export",
    }
