from __future__ import annotations

from typing import Any

from SyMBac.PSF import Camera, PSF_generator
from SyMBac.config_models import RenderConfig
from SyMBac.renderer import RenderResult, Renderer


class RendererService:
    def build_psf(self, params: dict[str, Any]) -> PSF_generator:
        psf = PSF_generator(**params)
        psf.calculate_PSF()
        return psf

    def build_camera(self, params: dict[str, Any] | None) -> Camera | None:
        if params is None:
            return None
        return Camera(**params)

    def build_renderer(
        self,
        simulation,
        psf: PSF_generator,
        real_image,
        camera: Camera | None = None,
        additional_real_images=None,
    ) -> Renderer:
        return Renderer(
            simulation=simulation,
            PSF=psf,
            real_image=real_image,
            camera=camera,
            additional_real_images=additional_real_images,
        )

    def preview_frame(self, renderer: Renderer, frame_index: int, config: RenderConfig) -> RenderResult:
        return renderer.render_frame(frame_index=frame_index, config=config)
