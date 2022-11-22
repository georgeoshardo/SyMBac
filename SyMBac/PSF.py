import numpy as np
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.special import jv
import psfmodels as psfm
import warnings

class Camera:
    def __init__(self, baseline, sensitivity, dark_noise):
        self.baseline, self.sensitivity, self.dark_noise = baseline, sensitivity, dark_noise

    def render_dark_image(self, size, plot=True):
        rng = np.random.default_rng(2)
        dark_img = rng.normal(loc = self.baseline, scale=self.dark_noise, size=size)
        dark_img = rng.poisson(dark_img)
        if plot:
            plt.imshow(dark_img, cmap="Greys_r")
            plt.colorbar()
            plt.axis("off")
            plt.show()
        return dark_img


class PSF_generator:
    def __init__(self, radius, wavelength, NA, n, apo_sigma, mode, condenser=None, z_height=None, resize_amount=None, pix_mic_conv=None, scale=None):
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
            self.W, self.R, self.diameter = get_condensers()[condenser]

        self.z_height = z_height
        self.min_sigma = 0.42 * 0.6 / 6 / self.scale  # micron#

    def calculate_PSF(self):
        if "phase contrast" in self.mode.lower():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.kernel = get_phase_contrast_kernel(R=self.R, W=self.W, radius=self.radius, scale=self.scale, NA=self.NA, n=self.n, sigma=self.apo_sigma, wavelength=self.wavelength)

        elif "simple fluo" in self.mode.lower():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.kernel = get_fluorescence_kernel(radius=self.radius, scale=self.scale, NA=self.NA, n=self.n, wavelength=self.wavelength)

        elif "3d fluo" in self.mode.lower():
            assert self.z_height, "For 3D fluorescence, you must specify a Z height"
            self.kernel = psfm.make_psf(self.z_height, self.radius*2, dxy=self.scale, dz=self.scale, pz=0, ni=self.n, wvl=self.wavelength, NA = self.NA)

        else:
            raise NameError("Incorrect mode, currently supported: phase contrast, simple fluo, 3d flup")

    def plot_PSF(self):
        if "3d fluo" in self.mode.lower():
            fig, axes = plt.subplots(1, 3)
            for dim, ax in enumerate(axes.flatten()):
                ax.axis("off")
                ax.imshow((self.kernel.mean(axis=dim)))
                scalebar = ScaleBar(self.scale, "um", length_fraction=0.3)
                ax.add_artist(scalebar)
            plt.show()
        else:
            fig, ax = plt.subplots()
            ax.axis("off")
            ax.imshow(self.kernel)
            scalebar = ScaleBar(self.scale, "um", length_fraction=0.25)
            ax.add_artist(scalebar)
            plt.show()

def get_fluorescence_kernel(wavelength,NA,n,radius,scale):
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
    
    Returns
    -------
    2-D numpy array representing the fluorescence contrast PSF
    """
   

    r = np.arange(-radius,radius+1)
    kaw = 2* NA/n * np.pi/wavelength
    xx,yy = np.meshgrid(r,r)
    xx, yy = xx*scale, yy*scale
    rr = np.sqrt(xx**2+yy**2) * kaw 
    PSF = (2*jv(1,rr)/(rr))**2
    PSF[radius,radius] = 1
    return PSF

        
def somb(x):
    """
    Returns the sombrero function of a 2D numpy array.
    """
    z = np.zeros(x.shape)
    x = np.abs(x)
    idx = np.nonzero(x)
    z[idx] = 2*jv(1,np.pi*x[idx])/(np.pi*x[idx])
    return z



def get_phase_contrast_kernel(R,W,radius,scale,NA,n,sigma,wavelength):
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
    sigma : radius of a 2D gaussian of the same size as the PSF (in pixels) which is multiplied by the PSF to simulate apodisation of the PSF
    λ : The mean wavelength of the imaging light (in micron)

    Returns
    -------
    2-D numpy array representing the phase contrast PSF
    
    """
    scale1 = 1000 # micron per millimeter
    #F = F * scale1 # to microm
    Lambda = wavelength # in micron % wavelength of light
    R = R * scale1 # to microm
    W = W * scale1 # to microm
    #The corresponding point spread kernel function for the negative phase contrast 
    r = np.arange(-radius,radius+1)
    xx,yy = np.meshgrid(r,r)
    xx, yy = xx*scale, yy*scale
    kaw = 2* NA/n * np.pi/Lambda
    rr = np.sqrt(xx**2 + yy**2)*kaw

    kernel1 = 2*jv(1,rr)/(rr)
    kernel1[radius,radius] =  1

    kernel2 = 2*(R-W)**2/R**2 * jv(1,(R-W)**2/R**2 * rr)/rr
    kernel2[radius,radius] = np.nanmax(kernel2)

    kernel = kernel1 - kernel2
    kernel = kernel/np.max(kernel)
    kernel[radius,radius] =  1
    kernel = -kernel/np.sum(kernel)
    gaussian = gaussian_2D(radius*2+1, sigma)
    kernel = kernel * gaussian
    return kernel
    
def gaussian_2D(size, σ):
    """Returns a 2D gaussian (numpy array) of size (pixels x pixels) and gaussian radius (σ)"""
    x = np.linspace(0,size,size)
    μ = np.mean(x)
    A = 1/(σ*np.sqrt(2*np.pi))
    B = np.exp(-1/2 * (x-μ)**2/(σ**2))
    _gaussian_1D = A*B
    _gaussian_2D = np.outer(_gaussian_1D,_gaussian_1D)
    return _gaussian_2D

def get_condensers():
    """Returns a dictionary of common phase contrast condenser dimensions, where the numbers are W, R, diameter (in mm)"""
    condensers = {
    "Ph1": (0.45, 3.75, 24),
    "Ph2": (0.8, 5.0, 24),
    "Ph3": (1.0, 9.5, 24),
    "Ph4": (1.5, 14.0, 24),
    "PhF": (1.5, 19.0, 25)
    } #W, R, Diameter
    return condensers