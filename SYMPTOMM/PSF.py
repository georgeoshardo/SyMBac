import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jve
from matplotlib_scalebar.scalebar import ScaleBar

def somb(x):
    z = np.zeros(x.shape)
    x = np.abs(x)
    idx = np.nonzero(x)
    z[idx] = 2*jv(1,np.pi*x[idx])/(np.pi*x[idx])
    return z

def get_phase_contrast_kernel(R,W,radius,scale,F,resolution):
    scale1 = 1000 # micron per millimeter
    F = F * scale1 # to microm
    Lambda = 0.55 # in micron % wavelength of light
    R = R * scale1 # to microm
    W = W * scale1 # to microm
    #The corresponding point spread kernel function for the negative phase contrast 

    meshgrid_arrange = np.arange(-radius,radius + 1/resolution,1/resolution)
    [xx,yy] = np.meshgrid(meshgrid_arrange,meshgrid_arrange)
    rr = np.sqrt(xx**2 + yy**2)*scale
    rr_dl = rr*(1/F)*(1/Lambda); # scaling with F and Lambda for dimension correction
    kernel1 = np.pi*R**2*somb(2*R*rr_dl);     
    kernel2 = np.pi*(R-W)**2*somb(2*(R-W)*rr_dl)


    kernel = kernel1 - 0.5*kernel2
    #kernel = kernel/np.max(kernel)
    kernel[radius*resolution,radius*resolution] = kernel[radius*resolution,radius*resolution] + 1
    kernel = -kernel/np.sum(kernel)
    return kernel


def get_phase_contrast_kernel_yin(R,W,radius,scale,F,resolution):
    scale1 = 1000 # micron per millimeter
    F = F * scale1 # to microm
    Lambda = 0.55 # in micron % wavelength of light
    R = R * scale1 # to microm
    W = W * scale1 # to microm
    #The corresponding point spread kernel function for the negative phase contrast 

    meshgrid_arrange = np.arange(-radius,radius + 1/resolution,1/resolution)
    [xx,yy] = np.meshgrid(meshgrid_arrange,meshgrid_arrange)
    rr = np.sqrt(xx**2 + yy**2)*scale
    rr_dl = rr*(1/F)*(1/Lambda); # scaling with F and Lambda for dimension correction
    kernel1 = np.pi*R*somb(2*R*rr_dl);     
    kernel2 = np.pi*(R-W)*somb(2*(R-W)*rr_dl)


    kernel = kernel1 - 0.5*kernel2
    #kernel = kernel/np.max(kernel)
    kernel[radius*resolution,radius*resolution] = kernel[radius*resolution,radius*resolution] + 1
    kernel = -kernel/np.sum(kernel)
    return kernel

def get_fluorescence_kernel(Lambda,NA,n,radius,resolution,scale):
    r = np.linspace(-radius,radius,int(radius *resolution)) 
    kaw = NA/n * 2 * np.pi/Lambda
    xx,yy = np.meshgrid(r,r) 
    rr = np.sqrt(xx**2+yy**2) * kaw *  scale 
    PSF = (2*jv(1,rr)/(rr))**2
    return PSF, rr

if __name__ == "__main__":
    kernel_type = "none"

    if kernel_type == "fluorescence":
        resolution = 4.5
        plt.figure(figsize=(3.1,3.1))
        radius = 30
        scale=0.16
        PSF = get_fluorescence_kernel(Lambda = 0.48, NA = 0.95, n = 1, radius = radius, resolution = resolution, scale=scale)
        plt.imshow((PSF)**(1/3),cmap="Greys_r")
        scalebar = ScaleBar(scale/resolution, 'um')
        plt.gca().add_artist(scalebar)
        plt.axis("off")
        #plt.savefig("psf.png",bbox_inches="tight",dpi=100)
        plt.show()
    else:
        F = 2
        W = 0.8
        R = 5
        resolution = 10
        scale =0.0625
        radius = 7
        kernel = get_phase_contrast_kernel(R,W,radius,scale,F,resolution)
        scalebar = ScaleBar(scale/resolution, 'um')
        plt.figure(figsize=(3.1,3.1))
        plt.gca().add_artist(scalebar)
        plt.imshow(kernel,cmap="Greys_r")
        plt.axis("off")
        plt.title("Phase Contrast")
        plt.savefig("Phase_PSF.png",bbox_inches="tight",dpi=300)
        plt.show()
        np.savetxt("/home/georgeos/kernel.txt",kernel)
