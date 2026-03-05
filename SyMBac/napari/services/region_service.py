from __future__ import annotations

import numpy as np


class RegionService:
    def auto_segment(self, renderer, classes: int = 3, cells: str = "dark") -> dict[str, np.ndarray]:
        regions = renderer.auto_segment_regions(classes=classes, cells=cells)
        return {
            "media": np.asarray(regions["media"], dtype=bool),
            "cell": np.asarray(regions["cell"], dtype=bool),
            "device": np.asarray(regions["device"], dtype=bool),
        }

    def apply_to_renderer(self, renderer, masks: dict[str, np.ndarray]) -> None:
        renderer.set_region_masks(
            media_mask=np.asarray(masks["media"], dtype=bool),
            cell_mask=np.asarray(masks["cell"], dtype=bool),
            device_mask=np.asarray(masks["device"], dtype=bool),
        )
