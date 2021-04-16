import numpy as np
from skimage.color import rgb2gray
from numpy import fft
from itertools import product
from PIL import Image
import copy
#from https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(phi, rho)
def pol2cart(phi,rho):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
def sfMatch(images,rescaling=0,tarmag=None):
    assert type(images) == type([]), 'The input must be a list.'

    numin = len(images)
    xs, ys = images[1].shape
    angs = np.zeros((xs,ys,numin))
    mags = np.zeros((xs,ys,numin))
    for x in range(numin):
        if len(images[x].shape) == 3:
            images[x] = rgb2gray(images[x])
        im1 = images[x]/255
        xs1, ys1 = im1.shape
        assert (xs == xs1) and (ys == ys1), 'All images must have the same size.'
        fftim1 = fft.fftshift(fft.fft2(im1))
        angs[:,:,x], mags[:,:,x] = cart2pol(np.real(fftim1),np.imag(fftim1))

    if type(tarmag) == type(None):
        tarmag = np.mean(mags,2)

    xt, yt = tarmag.shape
    assert (xs == xt) and (ys == yt), 'The target spectrum must have the same size as the images.'
    f1 = np.linspace(-ys/2, ys/2-1,ys)
    f2 = np.linspace(-xs/2, xs/2-1,xs)
    XX, YY = np.meshgrid(f1,f2)
    t, r = cart2pol(XX,YY)
    if xs%2 == 1 or ys%2 == 1:
        r = np.round(r) - 1
    else:
        r = np.round(r)
    output_images = []
    for x in range(numin):
        fftim = mags[:,:,x]
        a = fftim.T.ravel()
        accmap = r.T.ravel()+1
        a2 = tarmag.T.ravel()
        en_old = np.array([np.sum([a[x] for x in y]) for y in [list(np.where(accmap==z)) for z in np.unique(accmap).tolist()]])
        en_new = np.array([np.sum([a2[x] for x in y]) for y in [list(np.where(accmap==z)) for z in np.unique(accmap).tolist()]])
        coefficient = en_new/en_old
        cmat = coefficient[(r).astype(int)]# coefficient[r+1]
        cmat[r>np.floor(np.max((xs,ys))/2)] = 0
        newmag = fftim*cmat
        XX, YY = pol2cart(angs[:,:,x],newmag)
        new = XX+YY*complex(0,1)
        output = np.real(fft.ifft2(fft.ifftshift(new)))
        if rescaling == 0:
            output = (output*255).astype(np.uint8)
        output_images.append(output)
    if rescaling != 0:
        output_images = rescale(output_images,rescaling)
    return output_images

def rescale(images, option = 1):
    assert type(images) == type([]), 'The input must be a list.'
    assert option == 1 or option == 2, "Invalid rescaling option"
    numin = len(images)
    brightests = np.zeros((numin,1))
    darkests = np.zeros((numin,1))
    for n in range(numin):
        if len(images[n].shape) == 3:
            images[n] = rgb2gray(images[n])
        brightests[n] = np.max(images[n])
        darkests[n] = np.min(images[n])
    the_brightest = np.max(brightests)
    the_darkest = np.min(darkests)
    avg_brightest = np.mean(brightests)
    avg_darkest = np.mean(darkests)
    output_images = []
    for m in range(numin):
        if option == 1:
            rescaled = (images[m] - the_darkest)/(the_brightest - the_darkest)*255
        elif option == 2:
            rescaled = (images[m] - avg_darkest)/(avg_brightest - avg_darkest)*255
        output_images.append(rescaled.astype(np.uint8))
    return output_images

def lumMatch(images, mask = None, lum = None):
    assert type(images) == type([]), 'The input must be a list.'
    assert (mask == None) or  type(mask) == type([]), 'The input mask must be a list.'
    

    numin = len(images)
    if (mask == None) and (lum == None):
        print("HI")
        M = 0; S = 0
        for im in range(numin):
            if len(images[im].shape) == 3:
                images[im] = rgb2gray(images[im])
            M = M + np.mean(images[im])
            S = S + np.std(images[im])
        M = M/numin
        S = S/numin
        output_images = []
        for im in range(numin):
            im1 = copy.deepcopy(images[im])
            if np.std(im1) != 0:
                print("did_std")
                im1 = (im1 - np.mean(im1))/np.std(im1) * S + M
            else:
                im1[:,:] = M
            output_images.append(im1.astype(np.uint8))
    elif (mask != None) and (lum == None):
        print("HI2")
        M = 0; S = 0
        for im in range(numin):
            if len(images[im].shape) == 3:
                images[im] = rgb2gray(images[im])
            im1 = images[im]
            assert len(images) == len(mask), "The inputs must have the same length"
            m = mask[im]
            assert m.size == images[im].size, "Image and mask are not the same size"
            assert np.sum(m == 1) > 0, 'The mask must contain some ones.'
            M = M + np.mean(im1[m==1])
            S = S + np.mean(im1[m==1])
        M = M/numin
        S = S/numin
        output_images = []
        for im in range(numin):
            im1 = images[im]
            if type(mask) == type([]):
                m = mask[im] 
            if np.std(im1[m==1]):
                im1[m==1] = ( im1[m==1] - np.mean(im1[m==1]))/np.std(im1[m==1])* S + M
            else:
                im1[m==1] = M
            output_images.append(im1.astype(np.uint8))
    elif (mask != None) and (lum != None):
        print("HI3")
        M = lum[0]; S = lum[1]
        output_images = []
        for im in range(numin):
            if len(images[im].shape) == 3:
                images[im] = rgb2gray(images[im])
            im1 = images[im]
            if len(mask) == 0:
                if np.std(im1) != 0.0:
                    im1 = (im1 - np.mean(im1))/np.std(im1) * S + M
                else:
                    im1[:,:] = M
            else:
                if type(mask) == type([]):
                    assert len(images) == len(mask), "The inputs must have the same length"
                    m = mask[im]
                assert m.size == images[im].size, "Image and mask are not the same size"
                assert np.sum(m == 1) > 0, 'The mask must contain some ones.'
                if np.std(im1[m==1]) != 0.0:
                    im1[m==1] = (im1[m==1] - np.mean(im1[m==1]))/np.std(im1[m==1])*S + M
                else:
                    im1[m==1] = M
            output_images.append(im1)
    return output_images