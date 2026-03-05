from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _YamlModel(_StrictModel):
    schema_version: Literal["1.0"]
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path):
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)


class SimulationGeometrySpec(_StrictModel):
    trench_length: float
    trench_width: float
    pix_mic_conv: float
    resize_amount: int

    @model_validator(mode="after")
    def _validate(self):
        if self.trench_length <= 0:
            raise ValueError("trench_length must be > 0.")
        if self.trench_width <= 0:
            raise ValueError("trench_width must be > 0.")
        if self.pix_mic_conv <= 0:
            raise ValueError("pix_mic_conv must be > 0.")
        if self.resize_amount <= 0:
            raise ValueError("resize_amount must be > 0.")
        return self


class SimulationCellSpec(_StrictModel):
    cell_max_length: float
    cell_width: float
    max_length_std: float = 0.0
    width_std: float = 0.0
    width_upper_limit: float | None = None
    lysis_p: float = 0.0

    @model_validator(mode="after")
    def _validate(self):
        if self.cell_max_length <= 0:
            raise ValueError("cell_max_length must be > 0.")
        if self.cell_width <= 0:
            raise ValueError("cell_width must be > 0.")
        if self.max_length_std < 0:
            raise ValueError("max_length_std must be >= 0.")
        if self.width_std < 0:
            raise ValueError("width_std must be >= 0.")
        if self.width_upper_limit is not None and self.width_upper_limit <= 0:
            raise ValueError("width_upper_limit must be > 0 when provided.")
        if not (0 <= self.lysis_p <= 1):
            raise ValueError("lysis_p must be in [0, 1].")
        return self


class SimulationPhysicsSpec(_StrictModel):
    gravity: float
    phys_iters: int

    @model_validator(mode="after")
    def _validate(self):
        if self.phys_iters <= 0:
            raise ValueError("phys_iters must be > 0.")
        return self


class SimulationRuntimeSpec(_StrictModel):
    sim_length: int
    substeps: int = 100
    save_dir: str
    load_sim_dir: str | None = None

    @model_validator(mode="after")
    def _validate(self):
        if self.sim_length <= 0:
            raise ValueError("sim_length must be > 0.")
        if self.substeps <= 0:
            raise ValueError("substeps must be > 0.")
        if not self.save_dir:
            raise ValueError("save_dir must be a non-empty path.")
        return self


class SimulationLowLevelSpec(_StrictModel):
    cell_config: dict[str, Any] | None = None
    physics_config: dict[str, Any] | None = None
    cell_config_overrides: dict[str, Any] = Field(default_factory=dict)
    physics_config_overrides: dict[str, Any] = Field(default_factory=dict)


class BrownianJitterSpec(_StrictModel):
    longitudinal_std: float = 0.0
    transverse_std: float = 0.0
    rotation_std: float = 0.0
    persistence: float = 0.85
    max_dx_fraction_of_trench_width: float = 0.20
    max_dy_fraction_of_segment_radius: float = 0.75
    max_dy_px_floor: float = 1.0
    max_dtheta: float = 0.03
    backoff_attempts: int = 5
    application_mode: Literal["teleport", "velocity", "impulse"] = "teleport"
    projection_angular_damping: float = 0.35

    @model_validator(mode="after")
    def _validate(self):
        for value, name in (
            (self.longitudinal_std, "longitudinal_std"),
            (self.transverse_std, "transverse_std"),
            (self.rotation_std, "rotation_std"),
        ):
            if value < 0:
                raise ValueError(f"brownian.{name} must be >= 0.")
        if not (0 <= self.persistence < 1):
            raise ValueError("brownian.persistence must be in [0, 1).")
        for value, name in (
            (self.max_dx_fraction_of_trench_width, "max_dx_fraction_of_trench_width"),
            (self.max_dy_fraction_of_segment_radius, "max_dy_fraction_of_segment_radius"),
            (self.max_dy_px_floor, "max_dy_px_floor"),
            (self.max_dtheta, "max_dtheta"),
        ):
            if value <= 0:
                raise ValueError(f"brownian.{name} must be > 0.")
        if self.backoff_attempts < 1:
            raise ValueError("brownian.backoff_attempts must be >= 1.")
        if not (0 <= self.projection_angular_damping <= 1):
            raise ValueError("brownian.projection_angular_damping must be in [0, 1].")
        return self


class SimulationSpec(_YamlModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["simulation_spec"] = "simulation_spec"
    geometry: SimulationGeometrySpec
    cell: SimulationCellSpec
    physics: SimulationPhysicsSpec
    runtime: SimulationRuntimeSpec
    low_level: SimulationLowLevelSpec = Field(default_factory=SimulationLowLevelSpec)
    brownian: BrownianJitterSpec = Field(default_factory=BrownianJitterSpec)


class RenderConfig(_YamlModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["render_config"] = "render_config"
    media_multiplier: float = 75.0
    cell_multiplier: float = 1.7
    device_multiplier: float = 29.0
    sigma: float = 8.85
    match_fourier: bool = False
    match_histogram: bool = True
    match_noise: bool = False
    noise_var: float = 0.001
    defocus: float = 3.0
    halo_top_intensity: float = 1.0
    halo_bottom_intensity: float = 1.0
    halo_start: float = 0.0
    halo_end: float = 1.0
    cell_texture_strength: float = 0.0
    cell_texture_scale: float = 70.0
    edge_floor_opl: float = 0.0

    @model_validator(mode="after")
    def _validate(self):
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0.")
        if self.noise_var < 0:
            raise ValueError("noise_var must be >= 0.")
        if self.defocus < 0:
            raise ValueError("defocus must be >= 0.")
        if self.cell_texture_strength < 0:
            raise ValueError("cell_texture_strength must be >= 0.")
        if self.cell_texture_scale <= 0:
            raise ValueError("cell_texture_scale must be > 0.")
        if self.edge_floor_opl < 0:
            raise ValueError("edge_floor_opl must be >= 0.")
        if not (0 <= self.halo_start <= 1):
            raise ValueError("halo_start must be in [0, 1].")
        if not (0 <= self.halo_end <= 1):
            raise ValueError("halo_end must be in [0, 1].")
        if self.halo_end < self.halo_start:
            raise ValueError("halo_end must be >= halo_start.")
        return self


class RandomDatasetPlan(_YamlModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["random_dataset_plan"] = "random_dataset_plan"
    burn_in: int = 0
    n_samples: int
    sample_amount: float = 0.02
    randomise_hist_match: bool = False
    randomise_noise_match: bool = False
    randomise_fourier_match: bool = False

    @model_validator(mode="after")
    def _validate(self):
        if self.burn_in < 0:
            raise ValueError("burn_in must be >= 0.")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be > 0.")
        if self.sample_amount < 0:
            raise ValueError("sample_amount must be >= 0.")
        return self


class TimeseriesDatasetPlan(_YamlModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["timeseries_dataset_plan"] = "timeseries_dataset_plan"
    burn_in: int = 0
    sample_amount: float = 0.02
    n_series: int = 1
    frames_per_series: int | None = None

    @model_validator(mode="after")
    def _validate(self):
        if self.burn_in < 0:
            raise ValueError("burn_in must be >= 0.")
        if self.sample_amount < 0:
            raise ValueError("sample_amount must be >= 0.")
        if self.n_series <= 0:
            raise ValueError("n_series must be > 0.")
        if self.frames_per_series is not None and self.frames_per_series <= 0:
            raise ValueError("frames_per_series must be > 0 when provided.")
        return self


DatasetPlan = Annotated[
    Union[RandomDatasetPlan, TimeseriesDatasetPlan],
    Field(discriminator="kind"),
]


class DatasetOutputConfig(_YamlModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["dataset_output_config"] = "dataset_output_config"
    save_dir: str
    image_format: Literal["png", "tif", "tiff"] = "tiff"
    mask_dtype: str = "uint16"
    n_jobs: int = 1
    prefix: str | None = None
    export_geff: bool = True

    @model_validator(mode="after")
    def _validate(self):
        if not self.save_dir:
            raise ValueError("save_dir must be a non-empty path.")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be non-zero.")
        return self


@dataclass(slots=True)
class RenderResult:
    image: Any
    mask: Any
    superres_mask: Any

