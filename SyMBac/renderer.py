import importlib

import numpy as np
import psfmodels as psfm

from SyMBac.drawing import make_images_same_shape, perc_diff
import warnings
import napari
import os
import skimage
import copy

from ipywidgets import interactive, fixed
from matplotlib import pyplot as plt
from skimage.transform import rescale, rotate
from skimage.util import random_noise
from joblib import Parallel, delayed
from tqdm import tqdm
from skimage import draw
from skimage.exposure import match_histograms, rescale_intensity
from scipy.ndimage import gaussian_filter
from numpy import fft
from PIL import Image
from SyMBac.PSF import PSF_generator
from SyMBac.pySHINE import cart2pol, sfMatch, lumMatch
from skimage.filters import threshold_multiotsu
import random

if importlib.util.find_spec("cupy") is None:
    from scipy.signal import convolve2d as cuconvolve

    njobs = -1
    warnings.warn("Could not load CuPy for SyMBac, are you using a GPU? Defaulting to CPU convolution.")


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
            Typically 1/resize_amount. So 1/3 will scale the image down by a factor of 3. We do this because we render the image and kernel at high resolution, so that we can do the convolution at high resolution.
        rescale_int : bool
            If True, rescale the intensities between 0 and 1 and return a float32 numpy array of the convolved downscaled image.

        Returns
        -------
        outupt : 2D numpy array
            The output of the convolution rescale operation
        """

        output = cuconvolve(image, kernel, mode="constant")
        # output = output.get()
        output = rescale(output, rescale_factor, anti_aliasing=False)

        if rescale_int:
            output = rescale_intensity(output.astype(np.float32), out_range=(0, 1))
        return output
else:
    import cupy as cp
    from cupyx.scipy.ndimage import convolve as cuconvolve

    njobs = 1


    def convolve_rescale(image, kernel, rescale_factor, rescale_int):
        """
        Convolves an image with a kernel, and rescales it to the correct size.

        Parameters
        ----------
        image : 2D numpy array
            The image
        kernel : 2D numpy array
            The kernel
        rescale_factor : int
            Typicall 1/resize_amount. So 1/3 will scale the image down by a factor of 3. We do this because we render the image and kernel at high resolution, so that we can do the convolution at high resolution.
        rescale_int : bool
            If True, rescale the intensities between 0 and 1 and return a float32 numpy array of the convolved downscaled image.

        Returns
        -------
        outupt : 2D numpy array
            The output of the convolution rescale operation
        """

        output = cuconvolve(cp.array(image), cp.array(kernel), mode="constant")
        output = output.get()
        output = rescale(output, rescale_factor, anti_aliasing=False)

        if rescale_int:
            output = rescale_intensity(output.astype(np.float32), out_range=(0, 1))
        return output


class Renderer:
    """
    Instantiates a renderer, which given a simulation, PSF, real image, and optionally a camera, generates the synthetic data

    Example:

    >>> from SyMBac.renderer import Renderer
    >>> my_renderer = Renderer(my_simulation, my_kernel, real_image, my_camera)
    >>> my_renderer.select_intensity_napari()
    >>> my_renderer.optimise_synth_image(manual_update=False)
    >>> my_renderer.generate_training_data(
            sample_amount=0.2,
            randomise_hist_match=True,
            randomise_noise_match=True,
            burn_in=40,
            n_samples = 500,
            save_dir="/tmp/test/",
            in_series=False
        )


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
        self.y_border_expansion_coefficient = 3
        self.x_border_expansion_coefficient = 3
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

    def select_intensity_napari(self, auto = True, classes = 3, cells = "dark"):
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



    def generate_test_comparison(self, media_multiplier=75, cell_multiplier=1.7, device_multiplier=29, sigma=8.85,
                                 scene_no=-1, match_fourier=False, match_histogram=True, match_noise=False,
                                 debug_plot=False, noise_var=0.001, defocus=3.0, halo_top_intensity = 1, halo_bottom_intensity = 1, halo_start = 0, halo_end = 1, random_real_image = None):
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



        Returns
        -------
        noisy_img : 2D numpy array
            The final simulated microscope image
        expanded_mask_resized_reshaped : 2D numpy array
            The final image's accompanying masks
        """

        expanded_scene, expanded_scene_no_cells, expanded_mask = self.generate_PC_OPL(
            scene=self.simulation.OPL_scenes[scene_no],
            mask=self.simulation.masks[scene_no],
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

        if self.PSF.mode == "phase contrast":
            self.PSF = PSF_generator(radius=self.PSF.radius, wavelength=self.PSF.wavelength, NA=self.PSF.NA,
                                     n=self.PSF.n, resize_amount=self.simulation.resize_amount,
                                     pix_mic_conv=self.simulation.pix_mic_conv, apo_sigma=sigma, mode="phase contrast",
                                     condenser=self.PSF.condenser)
            self.PSF.calculate_PSF()
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
                temp_conv = cuconvolve(cp.array(cells_3D[x]), cp.array(psf[x])).get()
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
            rng = np.random.default_rng(2)
            matched = matched / (matched.max() / self.real_image.max()) / sensitivity
            if match_fourier:
                matched += abs(matched.min()) # Preserve mean > 0 for rng.poisson(matched)
            matched = rng.poisson(matched)
            noisy_img = matched + rng.normal(loc=baseline, scale=dark_noise, size=matched.shape)
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
            segment_1_top_left = (
            0 + self.simulation.offset, int(self.simulation.main_segments.iloc[0]["bb"][0] + self.simulation.offset))
            segment_1_bottom_right = (
                int(self.simulation.main_segments.iloc[0]["bb"][3] + self.simulation.offset),
                int(self.simulation.main_segments.iloc[0]["bb"][2] + self.simulation.offset))

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
        if expanded_scene is None:
            self.simulation.main_segments = self.simulation.main_segments.reindex(
                index=self.simulation.main_segments.index[::-1])
            expanded_scene, expanded_scene_no_cells, expanded_mask = get_OPL_image(scene, mask,
                                                                                   media_multiplier, cell_multiplier,
                                                                                   device_multiplier,
                                                                                   y_border_expansion_coefficient,
                                                                                   x_border_expansion_coefficient,
                                                                                   defocus)
        return expanded_scene, expanded_scene_no_cells, expanded_mask

    def optimise_synth_image(self, manual_update):

        """

        :param bool manual_update: Whether to turn on manual updating. This is recommended if you have no/a slow GPU. Will display a button to allow manual updating of the image optimiser
        :return: ipywidget object for optimisation of synthetic data
        """

        self.real_media_mean = self.real_resize[np.where(self.media_label.data)].mean()
        self.real_cell_mean = self.real_resize[np.where(self.cell_label.data)].mean()
        self.real_device_mean = self.real_resize[np.where(self.device_label.data)].mean()
        self.real_means = np.array((self.real_media_mean, self.real_cell_mean, self.real_device_mean))

        self.real_media_var = self.real_resize[np.where(self.media_label.data)].var()
        self.real_cell_var = self.real_resize[np.where(self.cell_label.data)].var()
        self.real_device_var = self.real_resize[np.where(self.device_label.data)].var()
        self.real_vars = np.array((self.real_media_var, self.real_cell_var, self.real_device_var))

        self.image_params = (
        self.real_media_mean, self.real_cell_mean, self.real_device_mean, self.real_means, self.real_media_var,
        self.real_cell_var, self.real_device_var, self.real_vars)

        self.params = interactive(
            self.generate_test_comparison,
            {'manual': manual_update},
            media_multiplier=(-300, 300, 1),
            cell_multiplier=(-30, 30, 0.01),
            device_multiplier=(-300, 300, 1),
            sigma=(self.PSF.min_sigma, self.PSF.min_sigma * 20, self.PSF.min_sigma / 20),
            scene_no=(0, len(self.simulation.OPL_scenes) - 1, 1),
            noise_var=(0, 0.01, 0.0001),
            match_fourier=[True, False],
            match_histogram=[True, False],
            match_noise=[True, False],
            debug_plot=fixed(True),
            defocus=(0, 20, 0.1),
            halo_top_intensity = (0,1,0.1), 
            halo_bottom_intensity = (0,1,0.1),
            halo_start = (0,1,0.11),
            halo_end = (0,1,0.1),
            random_real_image=fixed(None),
        )

        return self.params

    def generate_training_data(self, sample_amount, randomise_hist_match, randomise_noise_match,
                               burn_in, n_samples, save_dir, in_series=False, seed=False, n_jobs = 1, dtype = np.uint8):
        """
        Generates the training data from a Jupyter interactive output of generate_test_comparison

        Parameters
        ----------
        sample_amount : float
            The percentage sampling variance (drawn from a uniform distribution) to vary intensities by. For example, a
            sample_amount of 0.05 will randomly sample +/- 5% above and below the chosen intensity for cells,
            media and device. Can be used to create a little bit of variance in the final training data.
        randomise_hist_match : bool
            If true, histogram matching is randomly turned on and off each time a training sample is generated
        randomise_noise_match : bool
            If true, noise matching is randomly turned on and off each time a training sample is generated
        burn_in : int
            Number of frames to wait before generating training data. Can be used to ignore the start of the simulation
            where the trench only has 1 cell in it.
        n_samples : int
            The number of training images to generate
        save_dir : str
            The save directory of the training data
        in_series : bool
            Whether the images should be randomly sampled, or rendered in the order that the simulation was run in.
        seed : float
            Optional arg, if specified then the numpy random seed will be set for the rendering, allows reproducible rendering results.

        """
        if seed:
            np.random.seed(seed)

        try:
            os.mkdir(save_dir)
        except:
            pass
        try:
            os.mkdir(save_dir + "/convolutions")
        except:
            pass
        try:
            os.mkdir(save_dir + "/masks")
        except:
            pass
        try:
            os.mkdir(save_dir + "/superres_masks")
        except:
            pass

        current_file_num = len(os.listdir(save_dir + "/convolutions"))

        if in_series:
            series_len = (self.simulation.sim_length) - burn_in
            n_series_to_sim = int(np.ceil(n_samples/series_len))

            media_multiplier_modifiers = np.repeat([np.random.uniform(1 - sample_amount, 1 + sample_amount) for _ in range(n_series_to_sim)], series_len)
            cell_multiplier_modifiers = np.repeat([np.random.uniform(1 - sample_amount, 1 + sample_amount) for _ in range(n_series_to_sim)], series_len)
            device_multiplier_modifiers = np.repeat([np.random.uniform(1 - sample_amount, 1 + sample_amount) for _ in range(n_series_to_sim)], series_len)
            sigma_modifiers = np.repeat([np.random.uniform(1 - sample_amount, 1 + sample_amount) for _ in range(n_series_to_sim)], series_len)

        else:
            media_multiplier_modifiers = [np.random.uniform(1 - sample_amount, 1 + sample_amount) for _ in range(n_samples)]
            cell_multiplier_modifiers = [np.random.uniform(1 - sample_amount, 1 + sample_amount) for _ in range(n_samples)]
            device_multiplier_modifiers = [np.random.uniform(1 - sample_amount, 1 + sample_amount) for _ in range(n_samples)]
            sigma_modifiers = [np.random.uniform(1 - sample_amount, 1 + sample_amount) for _ in range(n_samples)]


        def generate_samples(z, media_multiplier_modifier, cell_multiplier_modifier, device_multiplier_modifier, sigma_modifier):

            media_multiplier = media_multiplier_modifier * self.params.kwargs["media_multiplier"]
            cell_multiplier =  cell_multiplier_modifier * self.params.kwargs["cell_multiplier"]
            device_multiplier = device_multiplier_modifier * self.params.kwargs["device_multiplier"]
            sigma = sigma_modifier * self.params.kwargs["sigma"]
            
            if in_series:
                scene_no = (np.arange(burn_in, self.simulation.sim_length).tolist() * n_series_to_sim)[z] #burn_in + z % (self.simulation.sim_length - 2)
                # print(z, scene_no, media_multiplier)
                # can maybe re-run run_simulation and draw_scene when this loops back to 0
            else:
                scene_no = np.random.randint(burn_in, self.simulation.sim_length - 2)
            if randomise_hist_match:
                match_histogram = np.random.choice([True, False])
            else:
                match_histogram = self.params.kwargs["match_histogram"]
            if randomise_noise_match:
                match_noise = np.random.choice([True, False])
            else:
                match_noise = self.params.kwargs["match_noise"]
            
            if self.additional_real_images:
                random_real_image = random.choice(self.additional_real_images)
            else:
                random_real_image = None
            
            syn_image, mask, superres_mask = self.generate_test_comparison(
                media_multiplier=media_multiplier,
                cell_multiplier=cell_multiplier,
                device_multiplier=device_multiplier,
                sigma=sigma,
                scene_no=scene_no,
                match_fourier=self.params.kwargs["match_fourier"],
                match_histogram=match_histogram,
                match_noise=match_noise,
                debug_plot=False,
                noise_var=self.params.kwargs["noise_var"],
                defocus=self.params.kwargs["defocus"],
                halo_top_intensity = self.params.kwargs["halo_top_intensity"], 
                halo_bottom_intensity = self.params.kwargs["halo_bottom_intensity"],
                halo_start = self.params.kwargs["halo_start"],
                halo_end = self.params.kwargs["halo_end"],
                random_real_image = random_real_image
            )

            syn_image = Image.fromarray(skimage.img_as_uint(rescale_intensity(syn_image)))
            syn_image.save("{}/convolutions/synth_{}.tif".format(save_dir, str(z).zfill(5)))

            if (cell_multiplier == 0) or (cell_multiplier == 0.0):
                mask = np.zeros(mask.shape)
                mask = Image.fromarray(mask.astype(dtype))
                mask.save("{}/masks/synth_{}.tif".format(save_dir, str(z).zfill(5)))

                superres_mask = np.zeros(superres_mask.shape)
                superres_mask = Image.fromarray(superres_mask.astype(dtype))
                superres_mask.save("{}/superres_masks/synth_{}.tif".format(save_dir, str(z).zfill(5)))
            else:
                mask = Image.fromarray(mask.astype(dtype))
                mask.save("{}/masks/synth_{}.tif".format(save_dir, str(z).zfill(5)))

                superres_mask = Image.fromarray(superres_mask.astype(dtype))
                superres_mask.save("{}/superres_masks/synth_{}.tif".format(save_dir, str(z).zfill(5)))


        Parallel(n_jobs=n_jobs, backend="threading")(delayed(generate_samples)(z, media_multiplier_modifier, cell_multiplier_modifier, device_multiplier_modifier, sigma_modifier) for z, media_multiplier_modifier, cell_multiplier_modifier, device_multiplier_modifier, sigma_modifier in
                               tqdm(
                                zip(range(current_file_num, n_samples + current_file_num), media_multiplier_modifiers, cell_multiplier_modifiers, device_multiplier_modifiers, sigma_modifiers)
                                , desc="Sample generation"))
