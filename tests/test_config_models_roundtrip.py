from pydantic import ValidationError

import pytest

from SyMBac.config_models import (
    DatasetOutputConfig,
    RenderConfig,
    SimulationCellSpec,
    SimulationGeometrySpec,
    SimulationPhysicsSpec,
    SimulationRuntimeSpec,
    SimulationSpec,
    TimeseriesDatasetPlan,
)


def test_strict_models_reject_type_coercion_and_extra_fields():
    with pytest.raises(ValidationError):
        SimulationGeometrySpec(
            trench_length="15.0",
            trench_width=1.5,
            pix_mic_conv=0.065,
            resize_amount=1,
        )

    with pytest.raises(ValidationError):
        SimulationGeometrySpec(
            trench_length=15.0,
            trench_width=1.5,
            pix_mic_conv=0.065,
            resize_amount=1,
            unknown_field=123,
        )


def test_simulation_spec_yaml_roundtrip(tmp_path):
    spec = SimulationSpec(
        geometry=SimulationGeometrySpec(
            trench_length=15.0,
            trench_width=1.5,
            pix_mic_conv=0.065,
            resize_amount=1,
        ),
        cell=SimulationCellSpec(
            cell_max_length=6.0,
            cell_width=1.0,
            max_length_std=0.2,
            width_std=0.1,
            lysis_p=0.0,
        ),
        physics=SimulationPhysicsSpec(gravity=0.0, phys_iters=1),
        runtime=SimulationRuntimeSpec(
            sim_length=3,
            substeps=2,
            save_dir=str(tmp_path / "sim"),
        ),
    )

    path = tmp_path / "simulation_spec.yaml"
    spec.to_yaml(path)
    loaded = SimulationSpec.from_yaml(path)

    assert loaded.model_dump(mode="python") == spec.model_dump(mode="python")


def test_render_and_dataset_yaml_roundtrip(tmp_path):
    render_config = RenderConfig(media_multiplier=50.0, match_noise=True, edge_floor_opl=0.1)
    render_path = tmp_path / "render_config.yaml"
    render_config.to_yaml(render_path)
    loaded_render_config = RenderConfig.from_yaml(render_path)
    assert loaded_render_config.model_dump(mode="python") == render_config.model_dump(mode="python")

    plan = TimeseriesDatasetPlan(burn_in=2, sample_amount=0.05, n_series=3, frames_per_series=4)
    plan_path = tmp_path / "plan.yaml"
    plan.to_yaml(plan_path)
    loaded_plan = TimeseriesDatasetPlan.from_yaml(plan_path)
    assert loaded_plan.model_dump(mode="python") == plan.model_dump(mode="python")

    output = DatasetOutputConfig(
        save_dir=str(tmp_path / "dataset"),
        image_format="tiff",
        mask_dtype="uint16",
        n_jobs=2,
        export_geff=False,
    )
    output_path = tmp_path / "output.yaml"
    output.to_yaml(output_path)
    loaded_output = DatasetOutputConfig.from_yaml(output_path)
    assert loaded_output.model_dump(mode="python") == output.model_dump(mode="python")

    with pytest.raises(ValidationError):
        RenderConfig.model_validate({
            "schema_version": "1.0",
            "kind": "render_config",
            "unexpected": True,
        })


def test_dataset_output_config_defaults_disable_geff():
    output = DatasetOutputConfig(save_dir="/tmp/out")
    assert output.export_geff is False
