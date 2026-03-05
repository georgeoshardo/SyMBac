from __future__ import annotations

from pathlib import Path

from SyMBac.config_models import (
    DatasetOutputConfig,
    RandomDatasetPlan,
    RenderConfig,
    SimulationCellSpec,
    TimeseriesDatasetPlan,
)
from SyMBac.renderer import Renderer


class ExportService:
    def export_dataset(
        self,
        renderer: Renderer,
        plan: RandomDatasetPlan | TimeseriesDatasetPlan,
        output: DatasetOutputConfig,
        base_config: RenderConfig,
        seed: int | None = None,
    ) -> dict:
        return renderer.export_dataset(plan=plan, output=output, base_config=base_config, seed=seed)

    def export_batch_timeseries(
        self,
        *,
        base_spec,
        variants: list[dict[str, float]],
        renderer_service,
        simulation_service,
        real_image,
        psf_params,
        camera_params,
        plan: TimeseriesDatasetPlan,
        output: DatasetOutputConfig,
        base_config: RenderConfig,
        seed: int | None = None,
    ) -> list[dict]:
        output_root = Path(output.save_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        all_metadata: list[dict] = []
        for idx, variant in enumerate(variants):
            cell_data = base_spec.cell.model_dump(mode="python")
            cell_data.update(variant)
            variant_cell = SimulationCellSpec(**cell_data)

            sim_runtime = base_spec.runtime.model_copy(
                update={"save_dir": str(output_root / f"simulation_{idx:03d}")}
            )
            sim_spec = base_spec.model_copy(
                update={"cell": variant_cell, "runtime": sim_runtime},
                deep=True,
            )

            simulation = simulation_service.build(sim_spec)
            simulation_service.run(simulation, show_window=False)
            simulation_service.draw_opl(simulation, do_transformation=False, label_masks=True)

            psf = renderer_service.build_psf(psf_params)
            camera = renderer_service.build_camera(camera_params)
            renderer = renderer_service.build_renderer(
                simulation=simulation,
                psf=psf,
                real_image=real_image,
                camera=camera,
                additional_real_images=[real_image],
            )

            variant_output = output.model_copy(
                update={"save_dir": str(output_root / f"sim_{idx:03d}")}
            )
            metadata = renderer.export_dataset(
                plan=plan,
                output=variant_output,
                base_config=base_config,
                seed=None if seed is None else (seed + idx),
            )
            all_metadata.append(metadata)

        return all_metadata
