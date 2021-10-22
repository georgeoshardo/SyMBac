import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jve
from matplotlib_scalebar.scalebar import ScaleBar

def get_fluorescence_kernel(Lambda,NA,n,radius,scale):
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
    kaw = 2* NA/n * np.pi/Lambda
    xx,yy = np.meshgrid(r,r)
    xx, yy = xx*scale, yy*scale
    rr = np.sqrt(xx**2+yy**2) * kaw 
    PSF = (2*jv(1,rr)/(rr))**2
    PSF[radius,radius] = 1
    return PSF, np.sqrt(xx**2+yy**2)

        
def somb(x):
    """
    Returns the sombrero function of a 2D numpy array.
    """
    z = np.zeros(x.shape)
    x = np.abs(x)
    idx = np.nonzero(x)
    z[idx] = 2*jv(1,np.pi*x[idx])/(np.pi*x[idx])
    return z



def get_phase_contrast_kernel(R,W,radius,scale,NA,n,sigma,λ):
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
    Lambda = λ # in micron % wavelength of light
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



## our old PC kernel from Yin et al
#def get_phase_contrast_kernel(R,W,radius,scale,F,sigma,λ):
#    scale1 = 1000 # micron per millimeter
#    F = F * scale1 # to microm
#    Lambda = λ # in micron % wavelength of light
#    R = R * scale1 # to microm
#    W = W * scale1 # to microm
#    #The corresponding point spread kernel function for the negative phase contrast #

#    meshgrid_arrange = np.arange(-radius,radius + 1,1)
#    [xx,yy] = np.meshgrid(meshgrid_arrange,meshgrid_arrange)
#    rr = np.sqrt(xx**2 + yy**2)*scale
#    rr_dl = rr*(1/F)*(1/Lambda); # scaling with F and Lambda for dimension correction
#    kernel1 = np.pi*R**2*somb(2*R*rr_dl);     
#    kernel2 = np.pi*(R-W)**2*somb(2*(R-W)*rr_dl)


 #   kernel = kernel1 - kernel2
 #   kernel = kernel/np.max(kernel)
 #   kernel[radius,radius] = kernel[radius,radius] + 1
 #   kernel = -kernel/np.sum(kernel)
  #  gaussian = gaussian_2D(radius*2+1, sigma)
  #  kernel = kernel * gaussian
  #  return kernel
    
#This is the old definition. The new definition doesnt take the resolution parameter. Instead we scale the pixel size directly as this is easier to understand
#def get_fluorescence_kernel(Lambda,NA,n,radius,resolution,scale):
#    r = np.linspace(-radius,radius,int(radius *resolution)) 
#    kaw = NA/n * 2 * np.pi/Lambda
#    xx,yy = np.meshgrid(r,r) 
#    rr = np.sqrt(xx**2+yy**2) * kaw *  scale 
#    PSF = (2*jv(1,rr)/(rr))**2
#    return PSF, rr
