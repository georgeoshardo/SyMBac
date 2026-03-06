from __future__ import annotations

import numpy as np


# Layers that should be visible for each workflow tab.
TAB_VISIBLE_LAYERS: dict[str, tuple[str, ...]] = {
    "Simulation": ("OPL scenes", "Synthetic masks"),
    "Optics": ("Real image", "PSF preview"),
    "Regions": ("Real image", "Region media", "Region cell", "Region device"),
    "Tuning": ("Synthetic preview", "Preview mask", "Real image"),
    "Export": ("Synthetic preview", "Preview mask", "Real image"),
}


class LayerManager:
    def __init__(self, viewer):
        self.viewer = viewer

    def _find_layer(self, name: str):
        for layer in self.viewer.layers:
            if layer.name == name:
                return layer
        return None

    def upsert_image(self, name: str, data, **kwargs):
        layer = self._find_layer(name)
        if layer is None:
            return self.viewer.add_image(np.asarray(data), name=name, **kwargs)
        layer.data = np.asarray(data)
        return layer

    def upsert_labels(self, name: str, data, **kwargs):
        layer = self._find_layer(name)
        if layer is None:
            return self.viewer.add_labels(np.asarray(data), name=name, **kwargs)
        layer.data = np.asarray(data)
        return layer

    def show_only_layers(self, tab_name: str) -> None:
        visible = set(TAB_VISIBLE_LAYERS.get(tab_name, ()))
        managed = {name for names in TAB_VISIBLE_LAYERS.values() for name in names}
        for layer in self.viewer.layers:
            if layer.name in managed:
                layer.visible = layer.name in visible

    def update_simulation_layers(self, simulation) -> None:
        self.upsert_image("OPL scenes", np.asarray(simulation.OPL_scenes), colormap="gray")
        self.upsert_labels("Synthetic masks", np.asarray(simulation.masks))

    def update_real_image(self, real_image) -> None:
        self.upsert_image("Real image", np.asarray(real_image), colormap="gray")

    def update_psf_preview(self, kernel) -> None:
        data = np.asarray(kernel)
        if data.ndim == 3:
            data = data[data.shape[0] // 2]
        self.upsert_image("PSF preview", data, colormap="gray")

    def update_preview_layers(self, result) -> None:
        self.upsert_image("Synthetic preview", np.asarray(result.image), colormap="gray")
        self.upsert_labels("Preview mask", np.asarray(result.mask))

    def set_region_layers(self, masks: dict[str, np.ndarray]) -> None:
        self.upsert_labels("Region media", np.asarray(masks["media"], dtype=np.uint8))
        self.upsert_labels("Region cell", np.asarray(masks["cell"], dtype=np.uint8))
        self.upsert_labels("Region device", np.asarray(masks["device"], dtype=np.uint8))

    def collect_region_masks(self) -> dict[str, np.ndarray]:
        media = self._find_layer("Region media")
        cell = self._find_layer("Region cell")
        device = self._find_layer("Region device")
        if media is None or cell is None or device is None:
            raise ValueError("Region layers are missing. Run auto-segmentation first.")
        return {
            "media": np.asarray(media.data) > 0,
            "cell": np.asarray(cell.data) > 0,
            "device": np.asarray(device.data) > 0,
        }
