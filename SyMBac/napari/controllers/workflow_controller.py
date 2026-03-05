from __future__ import annotations

from SyMBac.config_models import (
    DatasetOutputConfig,
    RandomDatasetPlan,
    RenderConfig,
    SimulationSpec,
    TimeseriesDatasetPlan,
)
from SyMBac.napari.services.export_service import ExportService
from SyMBac.napari.services.region_service import RegionService
from SyMBac.napari.services.renderer_service import RendererService
from SyMBac.napari.services.simulation_service import SimulationService
from SyMBac.napari.state import NapariSessionState


class WorkflowController:
    def __init__(
        self,
        state: NapariSessionState,
        simulation_service: SimulationService | None = None,
        renderer_service: RendererService | None = None,
        region_service: RegionService | None = None,
        export_service: ExportService | None = None,
    ):
        self.state = state
        self.simulation_service = simulation_service or SimulationService()
        self.renderer_service = renderer_service or RendererService()
        self.region_service = region_service or RegionService()
        self.export_service = export_service or ExportService()

    def set_simulation_spec(self, spec: SimulationSpec) -> None:
        self.state.simulation_spec = spec
        self.state.simulation = None

    def run_simulation(self, show_window: bool = False):
        if self.state.simulation_spec is None:
            raise ValueError("Set a SimulationSpec before running simulation.")
        simulation = self.simulation_service.build(self.state.simulation_spec)
        self.simulation_service.run(simulation, show_window=show_window)
        self.state.simulation = simulation
        return simulation

    def draw_opl(self, do_transformation: bool = False, label_masks: bool = True):
        if self.state.simulation is None:
            raise ValueError("Run a simulation before drawing OPL scenes.")
        self.simulation_service.draw_opl(
            self.state.simulation,
            do_transformation=do_transformation,
            label_masks=label_masks,
        )
        return self.state.simulation

    def set_real_image(self, real_image) -> None:
        self.state.real_image = real_image

    def build_renderer(self, psf_params: dict, camera_params: dict | None = None):
        if self.state.simulation is None:
            raise ValueError("Simulation not available. Run and draw first.")
        if not hasattr(self.state.simulation, "OPL_scenes"):
            raise ValueError("Simulation has no OPL scenes yet. Run draw_opl first.")
        if self.state.real_image is None:
            raise ValueError("Real image is not set.")

        psf = self.renderer_service.build_psf(psf_params)
        camera = self.renderer_service.build_camera(camera_params)
        renderer = self.renderer_service.build_renderer(
            simulation=self.state.simulation,
            psf=psf,
            real_image=self.state.real_image,
            camera=camera,
            additional_real_images=[self.state.real_image],
        )
        self.state.psf_params = dict(psf_params)
        self.state.camera_params = dict(camera_params) if camera_params is not None else {}
        self.state.psf = psf
        self.state.camera = camera
        self.state.renderer = renderer
        return renderer

    def auto_segment_regions(self, classes: int = 3, cells: str = "dark"):
        if self.state.renderer is None:
            raise ValueError("Renderer has not been created.")
        return self.region_service.auto_segment(self.state.renderer, classes=classes, cells=cells)

    def set_region_masks(self, masks) -> None:
        if self.state.renderer is None:
            raise ValueError("Renderer has not been created.")
        self.region_service.apply_to_renderer(self.state.renderer, masks)

    def preview_frame(self, frame_index: int, config: RenderConfig):
        if self.state.renderer is None:
            raise ValueError("Renderer has not been created.")
        self.state.base_render_config = config
        return self.renderer_service.preview_frame(self.state.renderer, frame_index=frame_index, config=config)

    def export_dataset(
        self,
        plan: RandomDatasetPlan | TimeseriesDatasetPlan,
        output: DatasetOutputConfig,
        base_config: RenderConfig,
        seed: int | None = None,
    ):
        if self.state.renderer is None:
            raise ValueError("Renderer has not been created.")
        self.state.last_metadata = self.export_service.export_dataset(
            renderer=self.state.renderer,
            plan=plan,
            output=output,
            base_config=base_config,
            seed=seed,
        )
        return self.state.last_metadata

    def export_batch_timeseries(
        self,
        *,
        variants: list[dict[str, float]],
        plan: TimeseriesDatasetPlan,
        output: DatasetOutputConfig,
        base_config: RenderConfig,
        seed: int | None = None,
    ):
        if self.state.simulation_spec is None:
            raise ValueError("Set a SimulationSpec before batch export.")
        if self.state.real_image is None:
            raise ValueError("Real image is not set.")
        if not self.state.psf_params:
            raise ValueError("Build a renderer once so PSF parameters are available.")

        self.state.last_metadata = self.export_service.export_batch_timeseries(
            base_spec=self.state.simulation_spec,
            variants=variants,
            renderer_service=self.renderer_service,
            simulation_service=self.simulation_service,
            real_image=self.state.real_image,
            psf_params=self.state.psf_params,
            camera_params=self.state.camera_params or None,
            plan=plan,
            output=output,
            base_config=base_config,
            seed=seed,
        )
        return self.state.last_metadata
