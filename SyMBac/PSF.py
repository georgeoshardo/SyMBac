import numpy as np
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.special import jv
import psfmodels as psfm
import warnings


class Camera:
    """
    Class for instantiating Camera objects.

    Example:

    >>> my_camera = Camera(baseline=100, sensitivity=2.9, dark_noise=8)
    >>> my_camera.render_dark_image()

    """

    def __init__(self, baseline, sensitivity, dark_noise):
        """

        :param int baseline: The baseline intensity of the camera.
        :param float sensitivity: The camera sensitivity.
        :param dark_noise: The camera dark noise
        """
        self.baseline, self.sensitivity, self.dark_noise = baseline, sensitivity, dark_noise

    def render_dark_image(self, size, plot=True):
        """
        Render a sample synthetic dark image from the camera

        :param tuple(int, int) size: Size of the dark image.
        :param bool plot: Whether or not to plot the image.
        :return: Dark image sample.
        :rtype: np.ndarray
        """
        rng = np.random.default_rng(2)
        dark_img = rng.normal(loc=self.baseline, scale=self.dark_noise, size=size)
        dark_img = rng.poisson(dark_img)
        if plot:
            plt.imshow(dark_img, cmap="Greys_r")
            plt.colorbar()
            plt.axis("off")
            plt.show()
        return dark_img


class PSF_generator:
    """
    Instantiate a PSF generator, allows you to create phase contrast or fluorescence PSFs.

    Example:

    >>> #Creating a phase contrast PSF
    >>> my_kernel = PSF_generator(
            radius = 50,
            wavelength = 0.75,
            NA = 1.2,
            n = 1.3,
            resize_amount = 3,
            pix_mic_conv = 0.065,
            apo_sigma = 10,
            mode="phase contrast",
            condenser = "Ph3"
        )
    >>> my_kernel.calculate_PSF()
    >>> my_kernel.plot_PSF()

    """

    def __init__(self, radius, wavelength, NA, n, apo_sigma, mode, condenser=None, z_height=None, resize_amount=None,
                 pix_mic_conv=None, scale=None, offset = 0):
        """
        :param int radius: Radius of the PSF.
        :param float wavelength: Wavelength of imaging light in micron.
        :param float NA: Numerical aperture of the objective lens.
        :param float n: Refractive index of the imaging medium.
        :param float apo_sigma: Gaussian apodisation sigma for phase contrast PSF (will be ignored for fluorescence PSFs).
        :param str mode: Either ``phase contrast``, ``simple fluo``, or `3d fluo``.
        :param str condenser: Either ``Ph1``, ``Ph2``, ``Ph3``, ``Ph4``, or ``PhF`` (will be ignored for fluorescence PSFs).
        :param int z_height: The Z-size of a 3D fluorescence PSF. Will be ignored for ``mode=phase contrast`` or ``simple fluo``.
        :param int resize_amount: Upscaling factor, typically chosen to be 3.
        :param float pix_mic_conv: Micron per pixel conversion factor. E.g approx 0.1 for 60x on some cameras.
        :param float scale: If not provided will be calculated as ``self.pix_mic_conv / self.resize_amount``.
        :param float offset: A constant offset to add to the PSF, increases accuracy of long range effects, especially useful for colony simulations.``.
        """
        self.radius = radius
        self.wavelength = wavelength
        self.NA = NA
        self.n = n
        if scale:
            self.scale = scale
        else:
            self.resize_amount = resize_amount
            self.pix_mic_conv = pix_mic_conv
            self.scale = self.pix_mic_conv / self.resize_amount
        self.apo_sigma = apo_sigma
        self.mode = mode
        self.condenser = condenser
        if condenser:
            self.W, self.R, self.diameter = self.get_condensers()[condenser]

        self.z_height = z_height
        self.min_sigma = 0.42 * 0.6 / 6 / self.scale  # micron#
        self.offset = offset

    def calculate_PSF(self):
        if "phase contrast" in self.mode.lower():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.kernel = self.get_phase_contrast_kernel(R=self.R, W=self.W, radius=self.radius, scale=self.scale,
                                                             NA=self.NA, n=self.n, sigma=self.apo_sigma,
                                                             wavelength=self.wavelength, offset = self.offset)

        elif "simple fluo" in self.mode.lower():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.kernel = self.get_fluorescence_kernel(radius=self.radius, scale=self.scale, NA=self.NA, n=self.n,
                                                           wavelength=self.wavelength, offset = self.offset)

        elif "3d fluo" in self.mode.lower():
            assert self.z_height, "For 3D fluorescence, you must specify a Z height"
            self.kernel = psfm.make_psf(self.z_height, self.radius * 2, dxy=self.scale, dz=self.scale, pz=0, ni=self.n,
                                        wvl=self.wavelength, NA=self.NA) + self.offset

        else:
            raise NameError("Incorrect mode, currently supported: phase contrast, simple fluo, 3d flup")

    def plot_PSF(self):
        if "3d fluo" in self.mode.lower():
            fig, axes = plt.subplots(1, 3)
            for dim, ax in enumerate(axes.flatten()):
                ax.axis("off")
                ax.imshow((self.kernel.mean(axis=dim)), cmap="Greys_r")
                scalebar = ScaleBar(self.scale, "um", length_fraction=0.3)
                ax.add_artist(scalebar)
            plt.show()
        else:
            fig, ax = plt.subplots()
            ax.axis("off")
            ax.imshow(self.kernel, cmap="Greys_r")
            scalebar = ScaleBar(self.scale, "um", length_fraction=0.25)
            ax.add_artist(scalebar)
            ax.set_title(f"apo_sigma: {self.apo_sigma}")
            plt.show()

    @staticmethod
    def get_fluorescence_kernel(wavelength, NA, n, radius, scale, offset = 0):
        """
        Returns a 2D numpy array which is an airy-disk approximation of the fluorescence point spread function

        Parameters
        ----------
        Lambda : float
            Wavelength of imaging light (micron)
        NA : float
            Numerical aperture of the objective lens
        n : float
            Refractive index of the imaging medium (~1 for air, ~1.4-1.5 for oil)
        radius : int
            The radius of the PSF to be rendered in pixels
        scale : float
            The pixel size of the image to be rendered (micron/pix)
        offset : float
            A constant offset to add to the PSF, increases accuracy of long range effects, especially useful for colony simulations.

        Returns
        -------
        2-D numpy array representing the fluorescence contrast PSF
        """

        r = np.arange(-radius, radius + 1)
        kaw = 2 * NA / n * np.pi / wavelength #np.tan(np.arcsin(NA/n))
        xx, yy = np.meshgrid(r, r)
        xx, yy = xx * scale, yy * scale
        rr = np.sqrt(xx ** 2 + yy ** 2) * kaw
        PSF = (2 * jv(1, rr) / (rr)) ** 2
        PSF[radius, radius] = 1
        PSF += offset
        return PSF

    @staticmethod
    def somb(x):
        r"""
        Returns the sombrero function of a 2D numpy array, defined as

        .. math::
           somb(x)= \frac{2 J_1 (\pi x)}{\pi x}


        """
        z = np.zeros(x.shape)
        x = np.abs(x)
        idx = np.nonzero(x)
        z[idx] = 2 * jv(1, np.pi * x[idx]) / (np.pi * x[idx])
        return z

    @staticmethod
    def get_phase_contrast_kernel(R, W, radius, scale, NA, n, sigma, wavelength, offset = 0):
        """
        Returns a 2D numpy array which is the phase contrast kernel based on microscope parameters


        Parameters
        ----------
        R : float
            The radius of the phase contrast condenser (in mm)
        W : float
            The width of the phase contrast condenser opening (in mm)
        radius : int
            The radius of the PSF to be rendered in pixels
        scale : float
            The pixel size of the image to be rendered (micron/pix)
        NA : float
            Numerical aperture of the objective lens
        n : float
            Refractive index of the imaging medium (~1 for air, ~1.4-1.5 for oil)
        sigma : float
            radius of a 2D gaussian of the same size as the PSF (in pixels) which is multiplied by the PSF to simulate apodisation of the PSF
        λ : float
            The mean wavelength of the imaging light (in micron)
        offset : float
            A constant offset to add to the PSF, increases accuracy of long range effects, especially useful for colony simulations.

        Returns
        -------
        2-D numpy array representing the phase contrast PSF

        """
        scale1 = 1000  # micron per millimeter
        # F = F * scale1 # to microm
        Lambda = wavelength  # in micron % wavelength of light
        R = R * scale1  # to microm
        W = W * scale1  # to microm
        # The corresponding point spread kernel function for the negative phase contrast
        r = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(r, r)
        xx, yy = xx * scale, yy * scale
        kaw = 2 * NA / n * np.pi / Lambda
        rr = np.sqrt(xx ** 2 + yy ** 2) * kaw

        kernel1 = 2 * jv(1, rr) / (rr)
        kernel1[radius, radius] = 1

        kernel2 = 2 * (R - W) ** 2 / R ** 2 * jv(1, (R - W) ** 2 / R ** 2 * rr) / rr
        kernel2[radius, radius] = np.nanmax(kernel2)

        kernel = kernel1 - kernel2
        kernel = kernel / np.max(kernel)
        kernel[radius, radius] = 1
        kernel = -kernel / np.sum(kernel)
        gaussian = PSF_generator.gaussian_2D(radius * 2 + 1, sigma)
        kernel = kernel * gaussian
        kernel += offset
        return kernel

    @staticmethod
    def gaussian_2D(size, σ):
        """Returns a 2D gaussian (numpy array) of size (pixels x pixels) and gaussian radius (σ)"""
        x = np.linspace(0, size, size)
        μ = np.mean(x)
        A = 1 / (σ * np.sqrt(2 * np.pi))
        B = np.exp(-1 / 2 * (x - μ) ** 2 / (σ ** 2))
        _gaussian_1D = A * B
        _gaussian_2D = np.outer(_gaussian_1D, _gaussian_1D)
        return _gaussian_2D

    @staticmethod
    def get_condensers():
        """Returns a dictionary of common phase contrast condenser dimensions, where the numbers are W, R, diameter (in mm)"""
        condensers = {
            "Ph1": (0.45, 3.75, 24),
            "Ph2": (0.8, 5.0, 24),
            "Ph3": (1.0, 9.5, 24),
            "Ph4": (1.5, 14.0, 24),
            "PhF": (1.5, 19.0, 25)
        }  # W, R, Diameter
        return condensers
