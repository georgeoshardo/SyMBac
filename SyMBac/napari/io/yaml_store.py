from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml

from SyMBac.config_models import (
    DatasetOutputConfig,
    RandomDatasetPlan,
    RenderConfig,
    SimulationSpec,
    TimeseriesDatasetPlan,
)

ModelT = TypeVar("ModelT")


def model_to_yaml_text(model) -> str:
    return yaml.safe_dump(model.model_dump(mode="python"), sort_keys=False)


def model_from_yaml_text(model_cls: type[ModelT], text: str) -> ModelT:
    data = yaml.safe_load(text) or {}
    return model_cls.model_validate(data)


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(model_to_yaml_text(model))


def load_simulation_spec(path: str | Path) -> SimulationSpec:
    return SimulationSpec.from_yaml(path)


def load_render_config(path: str | Path) -> RenderConfig:
    return RenderConfig.from_yaml(path)


def load_output_config(path: str | Path) -> DatasetOutputConfig:
    return DatasetOutputConfig.from_yaml(path)


def load_dataset_plan(path: str | Path) -> RandomDatasetPlan | TimeseriesDatasetPlan:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    kind = data.get("kind")
    if kind == "random_dataset_plan":
        return RandomDatasetPlan.model_validate(data)
    if kind == "timeseries_dataset_plan":
        return TimeseriesDatasetPlan.model_validate(data)
    raise ValueError(f"Unsupported dataset plan kind: {kind!r}")
