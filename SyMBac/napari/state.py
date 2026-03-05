from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from SyMBac.config_models import (
    DatasetOutputConfig,
    RandomDatasetPlan,
    RenderConfig,
    SimulationSpec,
    TimeseriesDatasetPlan,
)


def _default_random_plan() -> RandomDatasetPlan:
    return RandomDatasetPlan(n_samples=100)


def _default_timeseries_plan() -> TimeseriesDatasetPlan:
    return TimeseriesDatasetPlan(n_series=1)


def _default_output() -> DatasetOutputConfig:
    return DatasetOutputConfig(save_dir="/tmp/symbac_napari")


@dataclass
class NapariSessionState:
    """Mutable state shared across SyMBac napari docks."""

    simulation_spec: SimulationSpec | None = None
    simulation: Any | None = None

    real_image: np.ndarray | None = None

    psf: Any | None = None
    camera: Any | None = None
    renderer: Any | None = None

    psf_params: dict[str, Any] = field(default_factory=dict)
    camera_params: dict[str, Any] = field(default_factory=dict)

    base_render_config: RenderConfig = field(default_factory=RenderConfig)
    random_plan: RandomDatasetPlan = field(default_factory=_default_random_plan)
    timeseries_plan: TimeseriesDatasetPlan = field(default_factory=_default_timeseries_plan)
    output_config: DatasetOutputConfig = field(default_factory=_default_output)

    workers: dict[str, Any] = field(default_factory=dict)
    last_metadata: dict[str, Any] | list[dict[str, Any]] | None = None
