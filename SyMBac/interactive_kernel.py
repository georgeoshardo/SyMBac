import numpy as np
from scipy import special
from scipy import signal
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from skimage.draw import draw
from matplotlib.widgets import Slider, Button, RadioButtons

def somb(x):
    z=np.zeros(np.shape(x));
    x = abs(x)
    idx = np.argwhere(x)
    z[idx] = special.jv(1,np.pi*x[idx])/(np.pi*x[idx]);
    return z
condensers = {
    "Ph1": (0.45, 3.75, 24),
    "Ph2": (0.8, 5.0, 24),
    "Ph3": (1.0, 9.5, 24),
    "Ph4": (1.5, 14.0, 24),
    "PhF": (1.5, 19.0, 25)
} #W, R, Diameter

def get_kernel(scale,radius,condenser):
    W, R, _ = condensers[condenser]
    R *= scale
    W *= scale
    diameter = 2*radius + 1
    xx,yy = np.meshgrid(range(-radius,radius), range(-radius,radius), sparse=True, indexing='ij')
    rr = np.sqrt(xx**2 + yy**2)
    kernel1 = np.pi * R**2 * somb(2*R*rr)
    kernel2 = np.pi * (R - W)**2 * somb(2*(R - W)*rr)
    kernel = np.nan_to_num(kernel1) - np.nan_to_num(kernel2)
    kernel = -kernel/np.linalg.norm(kernel)
    kernel[radius,radius] = kernel[radius,radius] + 1;
    kernel = -kernel
    return kernel

def plot_kernel(kernel):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle('Scale = {}'.format(scale))
    ax1.imshow(np.log(abs(kernel)))
    ax1.set_title("ln(abs(kernel))")
    ax2.imshow(kernel)
    ax2.set_title("Raw kernel")
    ax1.axis("off")
    ax2.axis("off")
    fig.tight_layout()
    if save == True:
        fig.savefig("../kernels/{}.jpeg".format(str(scale).zfill(4)))
        plt.show()
        plt.close("all")
    elif save == False:
        plt.show()

scale = 9000
dim = 100
mult=2.4
radius = 20
def get_image(scale=scale,dim=dim,mult=mult,radius=radius, condenser="Ph2"):
    radius = radius*scale/5000
    kernel = get_kernel(scale,10,condenser)
    img = np.zeros((dim, dim, 1), dtype=np.double)
    spacing = mult*radius
    for x in range(int(radius+radius/2), int(dim+radius), int(spacing)):
        for y in range(int(radius+radius/2),int(dim+radius), int(spacing)):
            rr, cc = draw.disk((x, y), radius, shape=img.shape)
            img[rr,cc] = 1
    img = img.reshape(dim,dim)
    convolved = signal.convolve2d(img, kernel,mode="same",boundary="fill")
    return convolved
fig, ax = plt.subplots()
l= plt.imshow(get_image())
plt.axis("off")
ax.margins(x=0)
axcolor = 'lightgoldenrodyellow'

axmult = plt.axes([0.1,0.0,0.65,0.03], facecolor=axcolor)
axscale = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor=axcolor)
axradius = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)
rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)

smult = Slider(axmult, 'mult', 1, 10, valinit=mult)
sscale = Slider(axscale, 'scale', 1, 10000, valinit=scale)
sradius = Slider(axradius, 'radius', 1, 50, valinit=radius)
rcondense = RadioButtons(rax, ('Ph1', 'Ph2', 'Ph3', "Ph4", "PhF"), active=1)

def update(var):
    mult = smult.val
    scale = sscale.val
    radius = sradius.val
    l.set_data(get_image(scale=sscale.val,dim=dim,mult=smult.val,radius=sradius.val))
    def PhFunc(label):
        l.set_data(get_image(scale=sscale.val,dim=dim,mult=smult.val,radius=sradius.val,condenser=label))
        fig.canvas.draw_idle()
    rcondense.on_clicked(PhFunc)
    fig.autoscale()
    fig.set_clim(vmin=new_vim, vmax=new_vmax)
    fig.canvas.draw_idle()



smult.on_changed(update)
sscale.on_changed(update)
sradius.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    smult.reset()
    sscale.reset()
    sradius.reset()
button.on_clicked(reset)

plt.show()
