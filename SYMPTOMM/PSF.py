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


#This is the old definition. The new definition doesnt take the resolution parameter. Instead we scale the pixel size directly as this is easier to understand
#def get_fluorescence_kernel(Lambda,NA,n,radius,resolution,scale):
#    r = np.linspace(-radius,radius,int(radius *resolution)) 
#    kaw = NA/n * 2 * np.pi/Lambda
#    xx,yy = np.meshgrid(r,r) 
#    rr = np.sqrt(xx**2+yy**2) * kaw *  scale 
#    PSF = (2*jv(1,rr)/(rr))**2
#    return PSF, rr

def get_fluorescence_kernel(Lambda,NA,n,radius,scale):
    r = np.arange(-radius,radius+1)
    kaw = 2* NA/n * np.pi/Lambda
    xx,yy = np.meshgrid(r,r)
    xx, yy = xx*scale, yy*scale
    rr = np.sqrt(xx**2+yy**2) * kaw 
    PSF = (2*jv(1,rr)/(rr))**2
    PSF[radius,radius] = 1
    return PSF, np.sqrt(xx**2+yy**2)

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
