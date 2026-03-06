import importlib
import numpy as np
import psfmodels as psfm
from pathlib import Path
import yaml

from SyMBac.drawing import make_images_same_shape, perc_diff, draw_scene, draw_scene_from_segments
import warnings
import skimage
import copy

from matplotlib import pyplot as plt
from skimage.transform import rescale
from skimage.util import random_noise
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from skimage import draw
from skimage.exposure import match_histograms, rescale_intensity
from scipy.ndimage import gaussian_filter
from numpy import fft
from PIL import Image
from SyMBac.PSF import PSF_generator
from SyMBac.pySHINE import cart2pol, sfMatch, lumMatch
from skimage.filters import threshold_multiotsu
from tifffile import imwrite
import random
from SyMBac.config_models import (
    DatasetOutputConfig,
    DatasetPlan,
    RandomDatasetPlan,
    RenderConfig,
    RenderResult,
    TimeseriesDatasetPlan,
)

_CONV_BACKEND = "fftconvolve"  # default (always available)

# Try CuPy first (NVIDIA)
if importlib.util.find_spec("cupy") is not None:
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import convolve as _cupy_convolve
        _CONV_BACKEND = "cupy"
    except ImportError:
        pass

# Try PyTorch (Apple Silicon MPS or NVIDIA CUDA)
if _CONV_BACKEND != "cupy" and importlib.util.find_spec("torch") is not None:
    try:
        import torch as _torch
        if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
            _CONV_BACKEND = "torch_mps"
        elif _torch.cuda.is_available():
            _CONV_BACKEND = "torch_cuda"
    except (ImportError, AttributeError):
        pass

if _CONV_BACKEND == "fftconvolve":
    from scipy.signal import fftconvolve as _fftconvolve
    warnings.warn(
        "No GPU backend (CuPy or PyTorch) found. "
        "Install CuPy (NVIDIA) or PyTorch (`pip install SyMBac[cupy]` or `pip install SyMBac[torch]`) "
        "to use GPU-accelerated convolution. Falling back to CPU FFT convolution."
    )


def _torch_convolve(image, kernel):
    """Convolve *image* with *kernel* using PyTorch (MPS or CUDA)."""
    device = "mps" if _CONV_BACKEND == "torch_mps" else "cuda"
    img_t = _torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    # Flip kernel for true convolution (conv2d does cross-correlation)
    kern_t = _torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    kern_t = _torch.flip(kern_t, [2, 3])
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    img_t = _torch.nn.functional.pad(img_t, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0)
    out = _torch.nn.functional.conv2d(img_t, kern_t)
    return out.squeeze().cpu().numpy()


def _convolve_2d(image, kernel):
    """Backend-agnostic 2D convolution (no rescale)."""
    if _CONV_BACKEND == "cupy":
        return _cupy_convolve(cp.array(image), cp.array(kernel), mode="constant").get()
    elif _CONV_BACKEND.startswith("torch"):
        return _torch_convolve(image, kernel)
    else:
        return _fftconvolve(image, kernel, mode="same")


def convolve_rescale(image, kernel, rescale_factor, rescale_int):
    """
    Convolves an image with a kernel, and rescales it to the correct size.

    Parameters
    ----------
    image : numpy.ndarray
        The image
    kernel : 2D numpy array
        The kernel
    rescale_factor : int
        Typically 1/resize_amount. So 1/3 will scale the image down by a
        factor of 3. We do this because we render the image and kernel at
        high resolution, so that we can do the convolution at high resolution.
    rescale_int : bool
        If True, rescale the intensities between 0 and 1 and return a
        float32 numpy array of the convolved downscaled image.

    Returns
    -------
    output : 2D numpy array
        The output of the convolution rescale operation
    """
    output = _convolve_2d(image, kernel)
    output = rescale(output, rescale_factor, anti_aliasing=False)

    if rescale_int:
        output = rescale_intensity(output.astype(np.float32), out_range=(0, 1))
    return output


class RenderTuner:
    def __init__(self, renderer, base_config: RenderConfig, manual_update: bool, initial_config: RenderConfig | None = None):
        self.renderer = renderer
        self.base_config = base_config
        self.initial_config = initial_config or base_config
        self._interactive = self._build_widget(manual_update=manual_update)

    @property
    def widget(self):
        return self._interactive

    def current_config(self) -> RenderConfig:
        kwargs = dict(self._interactive.kwargs)
        kwargs.pop("scene_no", None)
        kwargs.pop("debug_plot", None)
        kwargs.pop("random_real_image", None)
        return RenderConfig(**kwargs)

    def _build_widget(self, manual_update: bool):
        from ipywidgets import fixed, interactive

        edge_floor_range = self.renderer._estimate_edge_floor_slider_range()
        widget = interactive(
            self.renderer._render_frame_impl,
            {"manual": manual_update},
            media_multiplier=(-300, 300, 1),
            cell_multiplier=(-30, 30, 0.01),
            device_multiplier=(-300, 300, 1),
            sigma=(self.renderer.PSF.min_sigma, self.renderer.PSF.min_sigma * 20, self.renderer.PSF.min_sigma / 20),
            scene_no=(0, len(self.renderer.simulation.OPL_scenes) - 1, 1),
            noise_var=(0, 0.01, 0.0001),
            match_fourier=[True, False],
            match_histogram=[True, False],
            match_noise=[True, False],
            debug_plot=fixed(True),
            defocus=(0, 20, 0.1),
            halo_top_intensity=(0, 1, 0.1),
            halo_bottom_intensity=(0, 1, 0.1),
            halo_start=(0, 1, 0.1),
            halo_end=(0, 1, 0.1),
            random_real_image=fixed(None),
            cell_texture_strength=(0.0, 1.0, 0.05),
            cell_texture_scale=(10, 200, 1),
            edge_floor_opl=edge_floor_range,
        )
        values = self.initial_config.model_dump()
        for child in widget.children:
            name = getattr(child, "description", None)
            if name and name in values:
                try:
                    child.value = type(child.value)(values[name])
                except (ValueError, TypeError):
                    continue
        return widget


class Renderer:
    """
    Instantiates a renderer, which given a simulation, PSF, real image, and optionally a camera, generates the synthetic data

    Example:

    >>> from SyMBac.config_models import RenderConfig, TimeseriesDatasetPlan, DatasetOutputConfig
    >>> from SyMBac.renderer import Renderer
    >>> my_renderer = Renderer(my_simulation, my_kernel, real_image, my_camera)
    >>> my_renderer.select_intensity_napari()
    >>> tuner = my_renderer.create_tuner(base_config=RenderConfig(), manual_update=False)
    >>> plan = TimeseriesDatasetPlan(burn_in=40, n_series=2, frames_per_series=100, sample_amount=0.02)
    >>> output = DatasetOutputConfig(save_dir="/tmp/test_dataset")
    >>> my_renderer.export_dataset(plan=plan, output=output, base_config=tuner.current_config(), seed=42)


    """

    def __init__(self, simulation, PSF, real_image, camera=None, additional_real_images = None):
        """

        :param SyMBac.simulation.Simulation simulation: The SyMBac simulation.
        :param SyMBac.psf.PSF_generator PSF: The PSF to be applied to the synthetic data.
        :param np.ndarray real_image: A real image sample
        :param SyMBac.PSF.Camera camera: (optional) The simulation camera object to be applied to the synthetic data
        :param List additional_real_images: List of additional images which will be randomly used to fourier match during the rendering process.
        """
        self.real_image = real_image
        self.PSF = PSF
        self.simulation = simulation
        self.real_image = real_image
        self.camera = camera
        media_multiplier = 30
        cell_multiplier = 1
        device_multiplier = -50

        ## This bit is dedicated to figuring out the optimal render size
        sample_OPL = self.simulation.OPL_scenes[0]
        self.y_border_expansion_coefficient =  1 
        self.x_border_expansion_coefficient = 1
        self.additional_real_images = additional_real_images
        temp_expanded_scene, temp_expanded_scene_no_cells, temp_expanded_mask = self.generate_PC_OPL(
            scene=simulation.OPL_scenes[-1],
            mask=simulation.masks[-1],
            media_multiplier=media_multiplier,
            cell_multiplier=cell_multiplier,
            device_multiplier=cell_multiplier,
            y_border_expansion_coefficient=self.y_border_expansion_coefficient,
            x_border_expansion_coefficient=self.x_border_expansion_coefficient,
            defocus=30
        )




        self.y_border_expansion_coefficient = (np.ceil(real_image.shape[0] / temp_expanded_scene.shape[0]) * simulation.resize_amount) + 3
        self.x_border_expansion_coefficient = (np.ceil(real_image.shape[1] / temp_expanded_scene.shape[1]) * simulation.resize_amount) + 3


        temp_expanded_scene, temp_expanded_scene_no_cells, temp_expanded_mask = self.generate_PC_OPL(
            scene=simulation.OPL_scenes[-1],
            mask=simulation.masks[-1],
            media_multiplier=media_multiplier,
            cell_multiplier=cell_multiplier,
            device_multiplier=cell_multiplier,
            y_border_expansion_coefficient=self.y_border_expansion_coefficient,
            x_border_expansion_coefficient=self.x_border_expansion_coefficient,
            defocus=30
        )

        ### Generate temporary image to make same shape
        self.PSF.calculate_PSF()
        if "3d" in self.PSF.mode.lower():
            temp_kernel = self.PSF.kernel.mean(axis=0)
        else:
            temp_kernel = self.PSF.kernel

        convolved = convolve_rescale(temp_expanded_scene, temp_kernel, 1 / simulation.resize_amount, rescale_int=True)
        self.real_resize, self.expanded_resized = make_images_same_shape(real_image, convolved, rescale_int=True)
        mean_error = []
        media_error = []
        cell_error = []
        device_error = []

        mean_var_error = []
        media_var_error = []
        cell_var_error = []
        device_var_error = []

        self.error_params = (
        mean_error, media_error, cell_error, device_error, mean_var_error, media_var_error, cell_var_error,
        device_var_error)
        self._explicit_region_masks = None

    def select_intensity_napari(self, auto = True, classes = 3, cells = "dark"):
        import napari

        if auto:
            thresholds = threshold_multiotsu(self.real_resize, classes = classes)
            regions = np.digitize(self.real_resize, bins=thresholds)
            if cells == "dark":
                thresh_media = (regions == 2) * 1
                thresh_cells = (regions == 1) * 2
                thresh_device = (regions == 0) * 3
            elif cells == "light":
                thresh_media = ((regions == 0)) * 1
                thresh_cells = ((regions == 1)) * 2
                thresh_device = ((regions == 2)) * 3
        else:
            thresh_media = np.zeros(self.real_resize.shape).astype(int)
            thresh_cells = np.zeros(self.real_resize.shape).astype(int)
            thresh_device = np.zeros(self.real_resize.shape).astype(int)
        viewer = napari.view_image(self.real_resize)
        self.media_label = viewer.add_labels(thresh_media, name="Media")
        self.cell_label = viewer.add_labels(thresh_cells, name="Cell")
        self.device_label = viewer.add_labels(thresh_device, name="Device")
        self.set_region_masks(
            media_mask=np.asarray(thresh_media) > 0,
            cell_mask=np.asarray(thresh_cells) > 0,
            device_mask=np.asarray(thresh_device) > 0,
        )

    def set_region_masks(self, media_mask, cell_mask, device_mask) -> None:
        """
        Set explicit region masks used for image intensity parameter estimation.

        Parameters
        ----------
        media_mask, cell_mask, device_mask : 2D array-like
            Boolean-compatible masks matching ``self.real_resize.shape``.
        """
        shape = tuple(self.real_resize.shape)
        media = np.asarray(media_mask, dtype=bool)
        cell = np.asarray(cell_mask, dtype=bool)
        device = np.asarray(device_mask, dtype=bool)
        for name, mask in (("media", media), ("cell", cell), ("device", device)):
            if mask.shape != shape:
                raise ValueError(
                    f"{name}_mask shape {mask.shape} must match real image shape {shape}."
                )
        self._explicit_region_masks = {
            "media": media,
            "cell": cell,
            "device": device,
        }
        # Force recalculation with the new masks.
        self.image_params = None

    def clear_region_masks(self) -> None:
        """Clear explicit region masks and return to auto-segmentation/layer fallback."""
        self._explicit_region_masks = None
        self.image_params = None

    def auto_segment_regions(self, image=None, classes=3, cells="dark"):
        """
        Automatically segment an image into media, cell, and device regions
        using multi-Otsu thresholding. Returns region masks without requiring
        napari.

        Parameters
        ----------
        image : 2D numpy array, optional
            Image to segment. Defaults to self.real_resize.
        classes : int
            Number of Otsu classes.
        cells : str
            "dark" if cells are darker than media, "light" otherwise.

        Returns
        -------
        dict
            {"media": mask, "cell": mask, "device": mask} — boolean masks.
        """
        if image is None:
            image = self.real_resize
        thresholds = threshold_multiotsu(image, classes=classes)
        regions = np.digitize(image, bins=thresholds)
        if cells == "dark":
            return {
                "media": regions == 2,
                "cell": regions == 1,
                "device": regions == 0,
            }
        else:
            return {
                "media": regions == 0,
                "cell": regions == 1,
                "device": regions == 2,
            }

    def _ensure_image_params(self, cells="dark"):
        if hasattr(self, "image_params") and self.image_params is not None:
            return

        regions = None
        explicit = getattr(self, "_explicit_region_masks", None)
        if explicit is not None:
            regions = {
                "media": np.asarray(explicit["media"], dtype=bool),
                "cell": np.asarray(explicit["cell"], dtype=bool),
                "device": np.asarray(explicit["device"], dtype=bool),
            }
        else:
            # Fallback: if napari label layers exist from select_intensity_napari,
            # use those directly before auto-thresholding.
            media_layer = getattr(self, "media_label", None)
            cell_layer = getattr(self, "cell_label", None)
            device_layer = getattr(self, "device_label", None)
            if media_layer is not None and cell_layer is not None and device_layer is not None:
                media_data = np.asarray(getattr(media_layer, "data", media_layer))
                cell_data = np.asarray(getattr(cell_layer, "data", cell_layer))
                device_data = np.asarray(getattr(device_layer, "data", device_layer))
                if (
                    media_data.shape == self.real_resize.shape
                    and cell_data.shape == self.real_resize.shape
                    and device_data.shape == self.real_resize.shape
                ):
                    regions = {
                        "media": media_data > 0,
                        "cell": cell_data > 0,
                        "device": device_data > 0,
                    }

        if regions is None:
            regions = self.auto_segment_regions(image=self.real_resize, cells=cells)

        media = self.real_resize[regions["media"]]
        cell = self.real_resize[regions["cell"]]
        device = self.real_resize[regions["device"]]
        media_mean = float(media.mean()) if media.size else 0.0
        cell_mean = float(cell.mean()) if cell.size else 0.0
        device_mean = float(device.mean()) if device.size else 0.0
        media_var = float(media.var()) if media.size else 0.0
        cell_var = float(cell.var()) if cell.size else 0.0
        device_var = float(device.var()) if device.size else 0.0
        self.image_params = (
            media_mean,
            cell_mean,
            device_mean,
            np.array([media_mean, cell_mean, device_mean]),
            media_var,
            cell_var,
            device_var,
            np.array([media_var, cell_var, device_var]),
        )

    @staticmethod
    def _apply_edge_floor_opl(scene, edge_floor_opl):
        """
        Enforce a minimum non-zero OPL value for cell pixels.

        Parameters
        ----------
        scene : 2D numpy array
            Cell OPL scene where background is 0 and cells are >0.
        edge_floor_opl : float
            Minimum OPL for non-zero cell pixels. 0 disables this.

        Returns
        -------
        2D numpy array
            Scene with floor applied to non-zero cell pixels.
        """
        if edge_floor_opl is None or edge_floor_opl <= 0:
            return scene
        floored_scene = np.array(scene, copy=True)
        nonzero = floored_scene > 0
        floored_scene[nonzero] = np.maximum(floored_scene[nonzero], edge_floor_opl)
        return floored_scene

    def _estimate_edge_floor_slider_range(
        self,
        clamp_percentile=10.0,
        max_median_fraction=0.25,
        min_visible_fraction_of_max=0.8,
        burn_in_fraction=0.25,
        n_scenes=20,
        sample_per_scene=20000,
    ):
        """
        Estimate a safe interactive range for edge_floor_opl from simulation OPLs.

        The upper bound is based on a low non-zero OPL percentile, with an
        additional cap relative to median OPL to avoid over-flattening cells.
        A minimum fraction of max OPL is also enforced so the top of the slider
        produces a clearly visible effect.
        """
        default_range = (0.0, 20.0, 0.1)
        scenes = getattr(self.simulation, "OPL_scenes", None)
        if scenes is None or len(scenes) == 0:
            return default_range

        n_total = len(scenes)
        burn_in = int(n_total * burn_in_fraction)
        burn_in = max(0, min(burn_in, n_total - 1))
        available = n_total - burn_in
        if available <= 0:
            return default_range

        n_pick = min(n_scenes, available)
        idxs = np.unique(np.linspace(burn_in, n_total - 1, n_pick, dtype=int))
        rng = np.random.default_rng(0)
        sampled_nonzero = []
        for idx in idxs:
            scene = scenes[int(idx)]
            vals = scene[scene > 0]
            if vals.size == 0:
                continue
            if vals.size > sample_per_scene:
                vals = rng.choice(vals, size=sample_per_scene, replace=False)
            sampled_nonzero.append(vals.astype(np.float32, copy=False))

        if not sampled_nonzero:
            return default_range

        values = np.concatenate(sampled_nonzero)
        q = float(np.quantile(values, clamp_percentile / 100.0))
        med = float(np.median(values))
        upper = min(q, max_median_fraction * med)

        if not np.isfinite(upper) or upper <= 0:
            upper = q if np.isfinite(q) and q > 0 else float(values.max())
        if not np.isfinite(upper) or upper <= 0:
            return default_range

        max_val = float(values.max())
        # Keep range impactful: ensure max slider value is not tiny.
        visible_floor = float(min_visible_fraction_of_max * max_val)
        upper = float(max(upper, visible_floor))
        upper = float(min(upper, max_val))
        step = max(upper / 200.0, 0.001)
        upper = float(np.round(upper, 3))
        step = float(np.round(step, 4))
        return 0.0, upper, step

    def _rebuild_psf_for_sigma(self, sigma: float) -> None:
        if self.PSF.mode != "phase contrast":
            return

        self.PSF = PSF_generator(
            radius=self.PSF.radius,
            wavelength=self.PSF.wavelength,
            NA=self.PSF.NA,
            n=self.PSF.n,
            resize_amount=self.simulation.resize_amount,
            pix_mic_conv=self.simulation.pix_mic_conv,
            apo_sigma=sigma,
            mode="phase contrast",
            condenser=self.PSF.condenser,
            offset=getattr(self.PSF, "offset", 0),
        )
        self.PSF.calculate_PSF()

    def render_synthetic(self, media_multiplier=75, cell_multiplier=1.7,
                         device_multiplier=29, sigma=8.85, scene_no=-1,
                         match_fourier=False, match_histogram=True,
                         match_noise=False, noise_var=0.001, defocus=3.0,
                         halo_top_intensity=1, halo_bottom_intensity=1,
                         halo_start=0, halo_end=1, random_real_image=None,
                         edge_floor_opl=0.0):
        """
        Render a single synthetic image with the given parameters.

        This is a clean rendering path without side effects — no error list
        appending, no debug plots, no matplotlib. Suitable for automated
        optimization loops.

        Parameters
        ----------
        media_multiplier : float
            Intensity multiplier for media region.
        cell_multiplier : float
            Intensity multiplier for cell OPL.
        device_multiplier : float
            Intensity multiplier for device walls.
        sigma : float
            PSF apodisation sigma.
        scene_no : int
            Which simulation frame to render.
        match_fourier : bool
            Match rotational Fourier spectrum to real image.
        match_histogram : bool
            Match intensity histogram to real image.
        match_noise : bool
            Apply histogram match after noise addition.
        noise_var : float
            Gaussian noise variance (ad-hoc mode).
        defocus : float
            Defocus blur sigma applied to PSF kernel.
        halo_top_intensity : float
            Halo gradient top intensity.
        halo_bottom_intensity : float
            Halo gradient bottom intensity.
        halo_start : float
            Fractional position where halo ramp begins.
        halo_end : float
            Fractional position where halo ramp ends.
        random_real_image : 2D numpy array, optional
            Alternative real image for Fourier matching.
        edge_floor_opl : float
            Minimum non-zero OPL floor applied to cell pixels before
            multiplier scaling and convolution.

        Returns
        -------
        noisy_img : 2D numpy array
            The rendered synthetic image (float32, 0-1).
        mask : 2D numpy array
            Corresponding segmentation mask.
        """
        config = RenderConfig(
            media_multiplier=media_multiplier,
            cell_multiplier=cell_multiplier,
            device_multiplier=device_multiplier,
            sigma=sigma,
            match_fourier=match_fourier,
            match_histogram=match_histogram,
            match_noise=match_noise,
            noise_var=noise_var,
            defocus=defocus,
            halo_top_intensity=halo_top_intensity,
            halo_bottom_intensity=halo_bottom_intensity,
            halo_start=halo_start,
            halo_end=halo_end,
            edge_floor_opl=edge_floor_opl,
        )
        result = self.render_frame(
            frame_index=int(scene_no),
            config=config,
            real_image_override=random_real_image,
        )
        return result.image, result.mask

    def _render_frame_impl(self, media_multiplier=75, cell_multiplier=1.7, device_multiplier=29, sigma=8.85,
                           scene_no=-1, match_fourier=False, match_histogram=True, match_noise=False,
                           debug_plot=False, noise_var=0.001, defocus=3.0, halo_top_intensity = 1, halo_bottom_intensity = 1, halo_start = 0, halo_end = 1, random_real_image = None, cell_texture_strength=0.0, cell_texture_scale=70.0, edge_floor_opl=0.0):
        """
        Takes all the parameters we've defined and calculated, and uses them to finally generate a synthetic image.

        Parameters
        ----------
        media_multiplier : float
            Intensity multiplier for media (the area between cells which isn't the device)
        cell_multiplier : float
            Intensity multiplier for cell
        device_multiplier : float
            Intensity multiplier for device
        sigma : float
            Radius of a gaussian which simulates PSF apodisation
        scene_no : int in range(len(cell_timeseries_properties))
            The index of which scene to render
        scale : float
            The micron/pixel value of the image
        match_fourier : bool
            If true, use sfmatch to match the rotational fourier spectrum of the synthetic image to a real image sample
        match_histogram : bool
            If true, match the intensity histogram of a synthetic image to a real image
        offset : int
            The same offset value from draw_scene
        debug_plot : bool
            True if you want to see a quick preview of the rendered synthetic image
        noise_var : float
            The variance for the simulated camera noise (gaussian)
        kernel : SyMBac.PSF.PSF_generator
            A kernel object from SyMBac.PSF.PSF_generator
        resize_amount : int
            The upscaling factor to render the image by. E.g a resize_amount of 3 will interally render the image at 3x
            resolution before convolving and then downsampling the image. Values >2 are recommended.
        real_image : 2D numpy array
            A sample real image from the experiment you are trying to replicate
        image_params : tuple
            A tuple of parameters which describe the intensities and variances of the real image, in this order:
            (real_media_mean, real_cell_mean, real_device_mean, real_means, real_media_var, real_cell_var, real_device_var, real_vars).
        error_params : tuple
            A tuple of parameters which characterises the error between the intensities in the real image and the synthetic
            image, in this order: (mean_error,media_error,cell_error,device_error,mean_var_error,media_var_error,
            cell_var_error,device_var_error). I have given an example of their calculation in the example notebooks.
        fluorescence : bool
            If true converts image to a fluorescence (hides the trench and swaps to the fluorescence PSF).
        defocus : float
            Simulated optical defocus by convolving the kernel with a 2D gaussian of radius defocus.
        halo_top_intensity : float
            Simulated "halo" caused by the microfluidic device. This sets the starting muliplier of a linear ramp which is applied down the length of the image in the direction of the trench. , 
        halo_bottom_intensity : float
            Simulated "halo" caused by the microfluidic device. This sets the ending multiplier of a lienar ramp which is applied down the length of the image. E.g, if ``image`` has shape ``(y, x)``, then this results in ``image = image * np.linspace(halo_lower_int,halo_upper_int, image.shape[0])[:, None]``.
        edge_floor_opl : float
            Minimum non-zero OPL floor applied to cell pixels before
            multiplier scaling and convolution.



        Returns
        -------
        noisy_img : 2D numpy array
            The final simulated microscope image
        expanded_mask_resized_reshaped : 2D numpy array
            The final image's accompanying masks
        """
        self._ensure_image_params()

        base_scene = self.simulation.OPL_scenes[scene_no]

        if cell_texture_strength > 0:
            texture_params = {"strength": cell_texture_strength, "scale": cell_texture_scale}
            if hasattr(self.simulation, 'cell_timeseries_segments'):
                textured_scene, mask = draw_scene_from_segments(
                    self.simulation.cell_timeseries_segments[scene_no],
                    self.simulation._space_size,
                    self.simulation.offset,
                    self.simulation._label_masks,
                    cell_texture=texture_params,
                )
            else:
                textured_scene, mask = draw_scene(
                    self.simulation.cell_timeseries_properties[scene_no],
                    self.simulation._do_transformation,
                    self.simulation._space_size,
                    self.simulation.offset,
                    self.simulation._label_masks,
                    cell_texture=texture_params,
                )
            # Apply OPL floor on the untextured scene, then reapply texture
            # modulation to preserve texture contrast at higher floor values.
            if edge_floor_opl > 0 and base_scene.shape == textured_scene.shape:
                floored_base_scene = self._apply_edge_floor_opl(base_scene, edge_floor_opl)
                texture_factor = np.ones_like(base_scene, dtype=np.float32)
                np.divide(
                    textured_scene,
                    base_scene,
                    out=texture_factor,
                    where=base_scene > 1e-6,
                )
                texture_factor = np.clip(texture_factor, 0.0, 2.5)
                scene = floored_base_scene * texture_factor
            else:
                scene = self._apply_edge_floor_opl(textured_scene, edge_floor_opl)
        else:
            scene = self._apply_edge_floor_opl(base_scene, edge_floor_opl)
            mask = self.simulation.masks[scene_no]

        expanded_scene, expanded_scene_no_cells, expanded_mask = self.generate_PC_OPL(
            scene=scene,
            mask=mask,
            media_multiplier=media_multiplier,
            cell_multiplier=cell_multiplier,
            device_multiplier=device_multiplier,
            x_border_expansion_coefficient=self.x_border_expansion_coefficient,
            y_border_expansion_coefficient=self.y_border_expansion_coefficient,
            defocus=defocus
        )

        ### Halo simulation
        def halo_line_profile(length, halo_top_intensity, halo_bottom_intensity, halo_start, halo_end):
            halo_start = int(halo_start * length)
            halo_end = int(halo_end * length)
            part_1 = np.linspace(halo_bottom_intensity,halo_bottom_intensity,   halo_start)
            part_2 = np.linspace(halo_bottom_intensity, halo_top_intensity, halo_end - halo_start)
            part_3 = np.linspace(halo_top_intensity, halo_top_intensity, length - halo_end)
            a = np.concatenate([part_1, part_2, part_3])[:, None]
            return a


        
        #halo_array = np.linspace(halo_lower_int,halo_upper_int, expanded_scene.shape[0])[:, None]
        halo_array = halo_line_profile(self.real_image.shape[0]*self.simulation.resize_amount, halo_top_intensity, halo_bottom_intensity, halo_start, halo_end)
        #halo_array[halo_start:] = 1
        expanded_scene[expanded_scene.shape[0] - len(halo_array):,:] *= halo_array
        expanded_scene_no_cells[expanded_scene_no_cells.shape[0] - len(halo_array):,:] *= halo_array

        if self.PSF.mode == "phase contrast":
            R, W, radius, scale, NA, n, _, λ = self.PSF.R, self.PSF.W, self.PSF.radius, self.PSF.scale, self.PSF.NA, self.PSF.n, self.PSF.apo_sigma, self.PSF.wavelength
        else:
            radius, scale, NA, n, _, λ = self.PSF.radius, self.PSF.scale, self.PSF.NA, self.PSF.n, self.PSF.apo_sigma, self.PSF.wavelength
        real_media_mean, real_cell_mean, real_device_mean, real_means, real_media_var, real_cell_var, real_device_var, real_vars = self.image_params
        mean_error, media_error, cell_error, device_error, mean_var_error, media_var_error, cell_var_error, device_var_error = self.error_params

        self._rebuild_psf_for_sigma(sigma)
        if self.PSF.mode.lower() == "3d fluo":  # Full 3D PSF model
            def generate_deviation_from_CL(centreline, thickness):
                return np.arange(thickness) + centreline - int(np.ceil(thickness / 2))

            def gen_3D_coords_from_2D(centreline, thickness):
                return np.where(test_cells == thickness) + (generate_deviation_from_CL(centreline, thickness),)

            volume_shape = expanded_scene.shape[0:] + (int(expanded_scene.max()),)
            test_cells = np.round(expanded_scene)
            centreline = int(expanded_scene.max() / 2)
            cells_3D = np.zeros(volume_shape)
            for t in range(int(expanded_scene.max() * 2)):
                test_coords = gen_3D_coords_from_2D(centreline, t)
                for x, y in zip(test_coords[0], (test_coords[1])):
                    for z in test_coords[2]:
                        cells_3D[x, y, z] = 1
            cells_3D = np.moveaxis(cells_3D, -1, 0)
            psf = psfm.make_psf(volume_shape[2], radius * 2, dxy=scale, dz=scale, pz=0, ni=n, wvl=λ, NA=NA)
            convolved = np.zeros(cells_3D.shape)
            for x in range(len(cells_3D)):
                temp_conv = _convolve_2d(cells_3D[x], psf[x])
                convolved[x] = temp_conv
            convolved = convolved.sum(axis=0)
            convolved = rescale(convolved, 1 / self.simulation.resize_amount, anti_aliasing=False)
            convolved = rescale_intensity(convolved.astype(np.float32), out_range=(0, 1))
        else:
            kernel = self.PSF.kernel
            if defocus > 0:
                kernel = gaussian_filter(kernel, defocus, mode="reflect")
            convolved = convolve_rescale(expanded_scene, kernel, 1 / self.simulation.resize_amount, rescale_int=True)

        real_resize, expanded_resized = make_images_same_shape(self.real_image, convolved, rescale_int=True)
        if random_real_image is not None:
            fftim1 = fft.fftshift(fft.fft2(random_real_image))
        else:
            fftim1 = fft.fftshift(fft.fft2(real_resize))
            
        angs, mags = cart2pol(np.real(fftim1), np.imag(fftim1))

        if match_fourier and not match_histogram:
            matched = sfMatch([real_resize, expanded_resized], tarmag=mags)[1]
            matched = lumMatch([real_resize, matched], None, [np.mean(real_resize), np.std(real_resize)])[1]
        else:
            matched = expanded_resized

        if match_histogram and match_fourier:
            matched = sfMatch([real_resize, matched], tarmag=mags)[1]
            matched = lumMatch([real_resize, matched], None, [np.mean(real_resize), np.std(real_resize)])[1]
            matched = match_histograms(matched, real_resize)
        else:
            pass
        if match_histogram:
            matched = match_histograms(matched, real_resize)
        else:
            pass

        if self.camera:  # Camera noise simulation
            baseline, sensitivity, dark_noise = self.camera.baseline, self.camera.sensitivity, self.camera.dark_noise
            matched = matched / (matched.max() / self.real_image.max()) / sensitivity
            if match_fourier:
                matched += abs(matched.min()) # Preserve mean > 0 for rng.poisson(matched)
            matched = np.random.poisson(matched)
            noisy_img = matched + np.random.normal(loc=baseline, scale=dark_noise, size=matched.shape)
        else:  # Ad hoc noise mathcing
            noisy_img = random_noise(rescale_intensity(matched), mode="poisson")
            noisy_img = random_noise(rescale_intensity(noisy_img), mode="gaussian", mean=0, var=noise_var, clip=False)

        if match_noise:
            noisy_img = match_histograms(noisy_img, real_resize)
        else:
            pass
        noisy_img = rescale_intensity(noisy_img.astype(np.float32), out_range=(0, 1))

        ## getting the cell mask to the right shape
        expanded_mask_resized = rescale(expanded_mask, 1 / self.simulation.resize_amount, anti_aliasing=False,
                                        preserve_range=True,
                                        order=0)
        if len(np.unique(expanded_mask_resized)) > 2:
            _, expanded_mask_resized_reshaped = make_images_same_shape(self.real_image, expanded_mask_resized,
                                                                       rescale_int=False)
        else:
            _, expanded_mask_resized_reshaped = make_images_same_shape(self.real_image, expanded_mask_resized,
                                                                       rescale_int=True)

        expanded_media_mask = rescale(
            (expanded_scene_no_cells == device_multiplier) ^ (expanded_scene - expanded_scene_no_cells).astype(bool),
            1 / self.simulation.resize_amount, anti_aliasing=False)
        real_resize, expanded_media_mask = make_images_same_shape(self.real_image, expanded_media_mask,
                                                                  rescale_int=True)
        just_media = expanded_media_mask * noisy_img

        expanded_cell_pseudo_mask = (expanded_scene - expanded_scene_no_cells).astype(bool)
        expanded_cell_pseudo_mask = rescale(expanded_cell_pseudo_mask, 1 / self.simulation.resize_amount,
                                            anti_aliasing=False)

        real_resize, expanded_cell_pseudo_mask = make_images_same_shape(self.real_image, expanded_cell_pseudo_mask,
                                                                        rescale_int=True)
        just_cells = expanded_cell_pseudo_mask * noisy_img

        expanded_device_mask = expanded_scene_no_cells

        expanded_device_mask = rescale(expanded_device_mask, 1 / self.simulation.resize_amount, anti_aliasing=False)



        real_resize, expanded_device_mask = make_images_same_shape(self.real_image, expanded_device_mask,
                                                                   rescale_int=True)
        just_device = expanded_device_mask * noisy_img

        simulated_means = np.array([just_media[np.where(just_media)].mean(), just_cells[np.where(just_cells)].mean(),
                                    just_device[np.where(just_device)].mean()])
        simulated_vars = np.array([just_media[np.where(just_media)].var(), just_cells[np.where(just_cells)].var(),
                                   just_device[np.where(just_device)].var()])
        mean_error.append(perc_diff(np.mean(noisy_img), np.mean(real_resize)))
        mean_var_error.append(perc_diff(np.var(noisy_img), np.var(real_resize)))
        if "fluo" in self.PSF.mode.lower():
            pass
        else:
            media_error.append(perc_diff(simulated_means[0], real_media_mean))
            cell_error.append(perc_diff(simulated_means[1], real_cell_mean))
            device_error.append(perc_diff(simulated_means[2], real_device_mean))

            media_var_error.append(perc_diff(simulated_vars[0], real_media_var))
            cell_var_error.append(perc_diff(simulated_vars[1], real_cell_var))
            device_var_error.append(perc_diff(simulated_vars[2], real_device_var))
        if debug_plot:
            fig = plt.figure(figsize=(15, 5))
            ax1 = plt.subplot2grid((1, 8), (0, 0), colspan=1, rowspan=1)
            ax2 = plt.subplot2grid((1, 8), (0, 1), colspan=1, rowspan=1)
            ax3 = plt.subplot2grid((1, 8), (0, 2), colspan=3, rowspan=1)
            ax4 = plt.subplot2grid((1, 8), (0, 5), colspan=3, rowspan=1)
            ax1.imshow(noisy_img, cmap="Greys_r")
            ax1.set_title("Synthetic")
            ax1.axis("off")
            ax2.imshow(real_resize, cmap="Greys_r")
            ax2.set_title("Real")
            ax2.axis("off")
            ax3.plot(mean_error)
            ax3.plot(media_error)
            ax3.plot(cell_error)
            ax3.plot(device_error)
            ax3.legend(["Mean error", "Media error", "Cell error", "Device error"])
            ax3.set_title("Intensity Error")
            ax3.hlines(0, ax3.get_xlim()[0], ax3.get_xlim()[1], color="k", linestyles="dotted")
            ax4.plot(mean_var_error)
            ax4.plot(media_var_error)
            ax4.plot(cell_var_error)
            ax4.plot(device_var_error)
            ax4.legend(["Mean error", "Media error", "Cell error", "Device error"])
            ax4.set_title("Variance Error")
            ax4.hlines(0, ax4.get_xlim()[0], ax4.get_xlim()[1], color="k", linestyles="dotted")
            fig.tight_layout()
            plt.show()
            plt.close()
        else:
            _, superres_mask = make_images_same_shape(np.zeros((self.real_image.shape[0]*self.simulation.resize_amount,self.real_image.shape[1]*self.simulation.resize_amount)), expanded_mask, rescale_int=False)
            return noisy_img, expanded_mask_resized_reshaped.astype(int), superres_mask.astype(int)
            #return noisy_img, expanded_mask_resized_reshaped.astype(int)

    def render_frame(self, frame_index: int, config: RenderConfig, real_image_override=None) -> RenderResult:
        if not isinstance(config, RenderConfig):
            raise TypeError("config must be a RenderConfig instance.")
        frame_index = self._normalize_frame_index(frame_index)
        self._ensure_image_params()
        image, mask, superres_mask = self._render_frame_impl(
            media_multiplier=config.media_multiplier,
            cell_multiplier=config.cell_multiplier,
            device_multiplier=config.device_multiplier,
            sigma=config.sigma,
            scene_no=int(frame_index),
            match_fourier=config.match_fourier,
            match_histogram=config.match_histogram,
            match_noise=config.match_noise,
            debug_plot=False,
            noise_var=config.noise_var,
            defocus=config.defocus,
            halo_top_intensity=config.halo_top_intensity,
            halo_bottom_intensity=config.halo_bottom_intensity,
            halo_start=config.halo_start,
            halo_end=config.halo_end,
            random_real_image=real_image_override,
            cell_texture_strength=config.cell_texture_strength,
            cell_texture_scale=config.cell_texture_scale,
            edge_floor_opl=config.edge_floor_opl,
        )
        return RenderResult(image=image, mask=mask, superres_mask=superres_mask)

    def generate_PC_OPL(self, scene, mask, media_multiplier, cell_multiplier, device_multiplier,
                        y_border_expansion_coefficient, x_border_expansion_coefficient, defocus):
        """
        Takes a scene drawing, adds the trenches and colours all parts of the image to generate a first-order phase contrast
        image, uncorrupted (unconvolved) by the phase contrat optics. Also has a fluorescence parameter to quickly switch to
        fluorescence if you want.

        Parameters
        ----------
        main_segments : list
            A list of the trench segments, used for drawing the trench
        offset : int
            The same offset from the draw_scene function. Used to know the cell offset.
        scene : 2D numpy array
            A scene image
        mask : 2D numpy array
            The mask for the scene
        media_multiplier : float
            Intensity multiplier for media (the area between cells which isn't the device)
        cell_multiplier : float
            Intensity multiplier for cell
        device_multiplier : float
            Intensity multiplier for device
        y_border_expansion_coefficient : int
            Another offset-like argument. Multiplies the size of the image on each side by this value. 3 is a good starting
            value because you want the image to be relatively larger than the PSF which you are convolving over it.
        x_border_expansion_coefficient : int
            Another offset-like argument. Multiplies the size of the image on each side by this value. 3 is a good starting
            value because you want the image to be relatively larger than the PSF which you are convolving over it.
        fluorescence : bool
            If true converts image to a fluorescence (hides the trench and swaps to the fluorescence PSF).
        defocus : float
            Simulated optical defocus by convolving the kernel with a 2D gaussian of radius defocus.

        Returns
        -------
        expanded_scene : 2D numpy array
            A large (expanded on x and y axis) image of cells in a trench, but unconvolved. (The raw PC image before
            convolution)
        expanded_scene_no_cells : 2D numpy array
            Same as expanded_scene, except with the cells removed (this is necessary for later intensity tuning)
        expanded_mask : 2D numpy array
            The masks for the expanded scene
        """

        def get_OPL_image(scene, mask, media_multiplier, cell_multiplier, device_multiplier,
                          y_border_expansion_coefficient, x_border_expansion_coefficient, defocus):
            segment_1_top_left = [
                0 + self.simulation.offset, int(self.simulation.main_segments.iloc[0]["bb"][0] + self.simulation.offset)
            ]

            

            segment_1_bottom_right = [
                int(self.simulation.main_segments.iloc[0]["bb"][3] + self.simulation.offset),
                int(self.simulation.main_segments.iloc[0]["bb"][2] + self.simulation.offset)
            ]

            segment_2_top_left = (
            0 + self.simulation.offset, int(self.simulation.main_segments.iloc[1]["bb"][0] + self.simulation.offset))
            segment_2_bottom_right = (
                int(self.simulation.main_segments.iloc[1]["bb"][3] + self.simulation.offset),
                int(self.simulation.main_segments.iloc[1]["bb"][2] + self.simulation.offset))

            if "fluo" in self.PSF.mode.lower():
                test_scene = np.zeros(scene.shape)
                media_multiplier = -1 * device_multiplier
            else:
                test_scene = np.zeros(scene.shape) + device_multiplier

                rr, cc = draw.rectangle(start=segment_1_top_left, end=segment_1_bottom_right, shape=test_scene.shape)
                test_scene[rr, cc] = 1 * media_multiplier




                rr, cc = draw.rectangle(start=segment_2_top_left, end=segment_2_bottom_right, shape=test_scene.shape)
                test_scene[rr, cc] = 1 * media_multiplier


                circ_midpoint_y = (segment_1_top_left[1] + segment_2_bottom_right[1]) / 2
                radius = (segment_1_top_left[1] - self.simulation.offset - (
                            segment_2_bottom_right[1] - self.simulation.offset)) / 2
                circ_midpoint_x = (self.simulation.offset) + radius



                rr, cc = draw.rectangle(start=segment_2_top_left, end=(circ_midpoint_x, segment_1_top_left[1]),
                                        shape=test_scene.shape)
                


                test_scene[rr.astype(int), cc.astype(int)] = 1 * media_multiplier

                
                rr, cc = draw.disk(center=(circ_midpoint_x, circ_midpoint_y), radius=radius, shape=test_scene.shape)
                rr_semi = rr[rr < (circ_midpoint_x + 1)]
                cc_semi = cc[rr < (circ_midpoint_x + 1)]



                test_scene[rr_semi, cc_semi] = device_multiplier
            no_cells = copy.deepcopy(test_scene)

            test_scene += scene * cell_multiplier
            if "fluo" in self.PSF.mode.lower():
                pass
            else:
                test_scene = np.where(no_cells != media_multiplier, test_scene, media_multiplier)
            test_scene = test_scene[segment_2_top_left[0]:segment_1_bottom_right[0],
                         segment_2_top_left[1]:segment_1_bottom_right[1]]

            mask = np.where(no_cells != media_multiplier, mask, 0)
            mask_resized = mask[segment_2_top_left[0]:segment_1_bottom_right[0],
                           segment_2_top_left[1]:segment_1_bottom_right[1]]

            no_cells = no_cells[segment_2_top_left[0]:segment_1_bottom_right[0],
                       segment_2_top_left[1]:segment_1_bottom_right[1]]
            expanded_scene_no_cells = np.zeros((int(no_cells.shape[0] * y_border_expansion_coefficient),
                                                int(no_cells.shape[
                                                        1] * x_border_expansion_coefficient))) + media_multiplier
            expanded_scene_no_cells[expanded_scene_no_cells.shape[0] - no_cells.shape[0]:,
            int(expanded_scene_no_cells.shape[1] / 2 - int(test_scene.shape[1] / 2)):int(
                expanded_scene_no_cells.shape[1] / 2 - int(test_scene.shape[1] / 2)) + no_cells.shape[1]] = no_cells
            if "fluo" in self.PSF.mode.lower():
                expanded_scene = np.zeros((int(test_scene.shape[0] * y_border_expansion_coefficient),
                                           int(test_scene.shape[1] * x_border_expansion_coefficient)))
                expanded_scene[expanded_scene.shape[0] - test_scene.shape[0]:,
                int(expanded_scene.shape[1] / 2 - int(test_scene.shape[1] / 2)):int(
                    expanded_scene.shape[1] / 2 - int(test_scene.shape[1] / 2)) + test_scene.shape[1]] = test_scene
            else:
                expanded_scene = np.zeros((int(test_scene.shape[0] * y_border_expansion_coefficient),
                                           int(test_scene.shape[
                                                   1] * x_border_expansion_coefficient))) + media_multiplier
                expanded_scene[expanded_scene.shape[0] - test_scene.shape[0]:,
                int(expanded_scene.shape[1] / 2 - int(test_scene.shape[1] / 2)):int(
                    expanded_scene.shape[1] / 2 - int(test_scene.shape[1] / 2)) + test_scene.shape[1]] = test_scene

            expanded_mask = np.zeros((int(test_scene.shape[0] * y_border_expansion_coefficient),
                                      int(test_scene.shape[1] * x_border_expansion_coefficient)))
            expanded_mask[expanded_mask.shape[0] - test_scene.shape[0]:,
            int(expanded_mask.shape[1] / 2 - int(test_scene.shape[1] / 2)):int(
                expanded_mask.shape[1] / 2 - int(test_scene.shape[1] / 2)) + test_scene.shape[1]] = mask_resized

            return expanded_scene, expanded_scene_no_cells, expanded_mask

        expanded_scene, expanded_scene_no_cells, expanded_mask = get_OPL_image(scene, mask,
                                                                               media_multiplier, cell_multiplier,
                                                                               device_multiplier,
                                                                               y_border_expansion_coefficient,
                                                                               x_border_expansion_coefficient,
                                                                               defocus)
        if expanded_scene is None or expanded_scene.size == 0:
            self.simulation.main_segments = self.simulation.main_segments.reindex(
                index=self.simulation.main_segments.index[::-1])
            expanded_scene, expanded_scene_no_cells, expanded_mask = get_OPL_image(scene, mask,
                                                                                   media_multiplier, cell_multiplier,
                                                                                   device_multiplier,
                                                                                   y_border_expansion_coefficient,
                                                                                   x_border_expansion_coefficient,
                                                                                   defocus)
        return expanded_scene, expanded_scene_no_cells, expanded_mask

    def create_tuner(
        self,
        base_config: RenderConfig,
        manual_update: bool = False,
        initial_config: RenderConfig | None = None,
        cells: str = "dark",
    ) -> RenderTuner:
        if not isinstance(base_config, RenderConfig):
            raise TypeError("base_config must be a RenderConfig instance.")
        if initial_config is not None and not isinstance(initial_config, RenderConfig):
            raise TypeError("initial_config must be a RenderConfig instance when provided.")
        self._ensure_image_params(cells=cells)
        return RenderTuner(
            renderer=self,
            base_config=base_config,
            manual_update=manual_update,
            initial_config=initial_config,
        )

    def _sample_render_config(self, base_config: RenderConfig, sample_amount: float, rng: np.random.Generator) -> RenderConfig:
        scale = lambda: float(rng.uniform(1 - sample_amount, 1 + sample_amount))
        return base_config.model_copy(
            update={
                "media_multiplier": base_config.media_multiplier * scale(),
                "cell_multiplier": base_config.cell_multiplier * scale(),
                "device_multiplier": base_config.device_multiplier * scale(),
                "sigma": base_config.sigma * scale(),
            }
        )

    @staticmethod
    def _save_image_pair(image, mask, image_path: Path, mask_path: Path, image_ext: str, mask_dtype: np.dtype) -> None:
        if image_ext == "png":
            Image.fromarray(skimage.img_as_uint(rescale_intensity(image))).save(image_path)
            Image.fromarray(mask.astype(mask_dtype, copy=False)).save(mask_path)
            return
        imwrite(image_path, skimage.img_as_uint(rescale_intensity(image)))
        imwrite(mask_path, mask.astype(mask_dtype, copy=False))

    def _scene_count(self) -> int:
        scenes = getattr(self.simulation, "OPL_scenes", None)
        if scenes is not None:
            return len(scenes)

        sim_length = getattr(self.simulation, "sim_length", None)
        if sim_length is None:
            raise ValueError("Simulation has no OPL scenes yet. Run draw_opl first.")
        return int(sim_length)

    def _normalize_frame_index(self, frame_index: int) -> int:
        total_frames = self._scene_count()
        normalized = int(frame_index)
        if normalized < 0:
            normalized += total_frames
        if normalized < 0 or normalized >= total_frames:
            raise IndexError(
                f"frame_index={frame_index} is out of range for {total_frames} renderable frames."
            )
        return normalized

    def _available_scene_indices(self, burn_in: int) -> np.ndarray:
        total_frames = self._scene_count()
        if burn_in < 0:
            raise ValueError("burn_in must be >= 0.")
        if burn_in >= total_frames:
            raise ValueError(
                f"burn_in={burn_in} leaves no frames to render from total_frames={total_frames}."
            )
        return np.arange(burn_in, total_frames, dtype=int)

    def export_dataset(
        self,
        plan: DatasetPlan,
        output: DatasetOutputConfig,
        base_config: RenderConfig,
        seed: int | None = None,
    ):
        if not isinstance(base_config, RenderConfig):
            raise TypeError("base_config must be a RenderConfig instance.")
        if not isinstance(output, DatasetOutputConfig):
            raise TypeError("output must be a DatasetOutputConfig instance.")
        if not isinstance(plan, (RandomDatasetPlan, TimeseriesDatasetPlan)):
            raise TypeError("plan must be a RandomDatasetPlan or TimeseriesDatasetPlan instance.")

        self._ensure_image_params()
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        output_dir = Path(output.save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_ext = "png" if output.image_format == "png" else "tiff"
        mask_dtype = np.dtype(output.mask_dtype)

        if isinstance(plan, RandomDatasetPlan):
            metadata = self._export_random_dataset(
                plan=plan,
                output=output,
                base_config=base_config,
                output_dir=output_dir,
                image_ext=image_ext,
                mask_dtype=mask_dtype,
                rng=rng,
            )
        else:
            metadata = self._export_timeseries_dataset(
                plan=plan,
                output=output,
                base_config=base_config,
                output_dir=output_dir,
                image_ext=image_ext,
                mask_dtype=mask_dtype,
                rng=rng,
            )

        metadata_path = output_dir / "metadata.yaml"
        with metadata_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(metadata, handle, sort_keys=False)
        return metadata

    def _export_random_dataset(
        self,
        plan: RandomDatasetPlan,
        output: DatasetOutputConfig,
        base_config: RenderConfig,
        output_dir: Path,
        image_ext: str,
        mask_dtype: np.dtype,
        rng: np.random.Generator,
    ):
        images_dir = output_dir / "images"
        masks_dir = output_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        self._available_scene_indices(plan.burn_in)
        frame_choices = rng.integers(
            low=plan.burn_in,
            high=self._scene_count(),
            size=plan.n_samples,
        )
        prefix = output.prefix or ""

        records = []
        for idx, frame_no in tqdm(
            enumerate(frame_choices),
            total=plan.n_samples,
            desc="Rendering random dataset",
        ):
            config = self._sample_render_config(base_config, plan.sample_amount, rng)
            config = config.model_copy(
                update={
                    "match_histogram": bool(rng.choice([True, False])) if plan.randomise_hist_match else config.match_histogram,
                    "match_noise": bool(rng.choice([True, False])) if plan.randomise_noise_match else config.match_noise,
                    "match_fourier": bool(rng.choice([True, False])) if plan.randomise_fourier_match else config.match_fourier,
                }
            )
            real_image_override = random.choice(self.additional_real_images) if self.additional_real_images else None
            result = self.render_frame(int(frame_no), config=config, real_image_override=real_image_override)
            image_path = images_dir / f"{prefix}sample_{idx:05d}.{image_ext}"
            mask_path = masks_dir / f"{prefix}sample_{idx:05d}.{image_ext}"
            self._save_image_pair(result.image, result.mask, image_path, mask_path, image_ext, mask_dtype)
            records.append(
                {
                    "sample_idx": int(idx),
                    "simulation_frame_idx": int(frame_no),
                    "image_path": str(image_path.relative_to(output_dir)),
                    "mask_path": str(mask_path.relative_to(output_dir)),
                    "render_config": config.model_dump(mode="python"),
                }
            )

        return {
            "schema_version": "1.0",
            "kind": "dataset_metadata",
            "dataset_kind": "random",
            "save_dir": str(output_dir.resolve()),
            "n_samples": int(plan.n_samples),
            "burn_in": int(plan.burn_in),
            "sample_amount": float(plan.sample_amount),
            "image_format": image_ext,
            "mask_dtype": str(mask_dtype),
            "export_geff": False,
            "samples": records,
        }

    def _export_timeseries_dataset(
        self,
        plan: TimeseriesDatasetPlan,
        output: DatasetOutputConfig,
        base_config: RenderConfig,
        output_dir: Path,
        image_ext: str,
        mask_dtype: np.dtype,
        rng: np.random.Generator,
    ):
        available_scene_nos = self._available_scene_indices(plan.burn_in)
        available_frames = len(available_scene_nos)
        frames_per_series = plan.frames_per_series or available_frames
        if frames_per_series > available_frames:
            raise ValueError(
                f"frames_per_series={frames_per_series} exceeds available frames={available_frames}."
            )
        scene_nos = available_scene_nos[:frames_per_series]
        prefix = output.prefix or ""

        metadata = {
            "schema_version": "1.0",
            "kind": "dataset_metadata",
            "dataset_kind": "timeseries",
            "save_dir": str(output_dir.resolve()),
            "n_series": int(plan.n_series),
            "burn_in": int(plan.burn_in),
            "frames_per_series": int(frames_per_series),
            "simulation_frame_start": int(scene_nos[0]),
            "simulation_frame_end_exclusive": int(scene_nos[-1] + 1),
            "sample_amount": float(plan.sample_amount),
            "image_format": image_ext,
            "mask_dtype": str(mask_dtype),
            "series": [],
        }

        for series_idx in range(plan.n_series):
            series_id = f"series_{series_idx:03d}"
            series_dir = output_dir / series_id
            images_dir = series_dir / "images"
            masks_dir = series_dir / "masks"
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)

            series_config = self._sample_render_config(base_config, plan.sample_amount, rng)
            series_real_image = random.choice(self.additional_real_images) if self.additional_real_images else None

            def _render_one(frame_idx, scene_no):
                result = self.render_frame(int(scene_no), config=series_config, real_image_override=series_real_image)
                image_path = images_dir / f"{prefix}frame_{frame_idx:05d}.{image_ext}"
                mask_path = masks_dir / f"{prefix}frame_{frame_idx:05d}.{image_ext}"
                self._save_image_pair(result.image, result.mask, image_path, mask_path, image_ext, mask_dtype)
                return {
                    "frame_idx": int(frame_idx),
                    "simulation_frame_idx": int(scene_no),
                    "image_path": str(image_path.relative_to(series_dir)),
                    "mask_path": str(mask_path.relative_to(series_dir)),
                }

            if output.n_jobs == 1:
                frames = [_render_one(frame_idx, scene_no) for frame_idx, scene_no in enumerate(scene_nos)]
            else:
                frames = Parallel(n_jobs=output.n_jobs, backend="threading")(
                    delayed(_render_one)(frame_idx, scene_no)
                    for frame_idx, scene_no in enumerate(scene_nos)
                )
                frames = sorted(frames, key=lambda item: item["frame_idx"])

            manifest = {
                "schema_version": "1.0",
                "kind": "series_manifest",
                "series_id": series_id,
                "simulation_frame_start": int(scene_nos[0]),
                "simulation_frame_end_exclusive": int(scene_nos[-1] + 1),
                "render_config": series_config.model_dump(mode="python"),
                "frames": frames,
            }
            manifest_path = series_dir / "manifest.yaml"
            with manifest_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(manifest, handle, sort_keys=False)

            geff_path = None
            if output.export_geff:
                from SyMBac.lineage import Lineage

                lineage = Lineage(self.simulation)
                geff_path = series_dir / "lineage.geff"
                lineage.to_geff(
                    str(geff_path),
                    frame_range=(int(scene_nos[0]), int(scene_nos[-1] + 1)),
                    overwrite=True,
                )

            metadata["series"].append(
                {
                    "series_id": series_id,
                    "manifest_path": str(manifest_path.relative_to(output_dir)),
                    "lineage_store": str(geff_path.relative_to(output_dir)) if geff_path else None,
                }
            )
        return metadata
