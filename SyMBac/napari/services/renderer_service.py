from __future__ import annotations

from typing import Any

from SyMBac.PSF import Camera, PSF_generator
from SyMBac.config_models import RenderConfig
from SyMBac.renderer import RenderResult, Renderer


class RendererService:
    @staticmethod
    def _normalize_psf_params(params: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise ValueError("PSF params must be a mapping.")

        normalized = dict(params)

        int_fields = ("radius", "resize_amount", "z_height", "pz")
        for field in int_fields:
            if field in normalized and normalized[field] is not None:
                normalized[field] = int(round(float(normalized[field])))

        float_fields = (
            "wavelength",
            "NA",
            "n",
            "apo_sigma",
            "pix_mic_conv",
            "scale",
            "offset",
            "working_distance",
        )
        for field in float_fields:
            if field in normalized and normalized[field] is not None:
                normalized[field] = float(normalized[field])

        # Basic sanity checks to fail early with clear user-facing messages.
        positive_fields = ("radius", "resize_amount", "wavelength", "NA", "n")
        for field in positive_fields:
            value = normalized.get(field)
            if value is None or value <= 0:
                raise ValueError(f"PSF '{field}' must be > 0.")

        if normalized.get("apo_sigma") is not None and normalized["apo_sigma"] <= 0:
            raise ValueError("PSF 'apo_sigma' must be > 0.")
        if normalized.get("pix_mic_conv") is not None and normalized["pix_mic_conv"] <= 0:
            raise ValueError("PSF 'pix_mic_conv' must be > 0.")

        mode = str(normalized.get("mode", "")).strip()
        if not mode:
            raise ValueError("PSF 'mode' is required.")
        normalized["mode"] = mode

        if "phase contrast" in mode.lower():
            condenser = normalized.get("condenser")
            if condenser is None:
                raise ValueError("PSF 'condenser' is required for phase contrast mode.")

        return normalized

    @staticmethod
    def _normalize_camera_params(params: dict[str, Any] | None) -> dict[str, float] | None:
        if params is None:
            return None
        if not isinstance(params, dict):
            raise ValueError("Camera params must be a mapping.")
        required = ("baseline", "sensitivity", "dark_noise")
        normalized: dict[str, float] = {}
        for field in required:
            if field not in params:
                raise ValueError(f"Camera '{field}' is required.")
            normalized[field] = float(params[field])

        if normalized["sensitivity"] <= 0:
            raise ValueError("Camera 'sensitivity' must be > 0.")
        if normalized["dark_noise"] < 0:
            raise ValueError("Camera 'dark_noise' must be >= 0.")
        return normalized

    def build_psf(self, params: dict[str, Any]) -> PSF_generator:
        psf = PSF_generator(**self._normalize_psf_params(params))
        psf.calculate_PSF()
        return psf

    def build_camera(self, params: dict[str, Any] | None) -> Camera | None:
        normalized = self._normalize_camera_params(params)
        if normalized is None:
            return None
        return Camera(**normalized)

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
