import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
import tifffile
import pkgutil
from io import BytesIO

from skimage.segmentation import find_boundaries
from skimage.transform import resize
from scipy.interpolate import RegularGridInterpolator


def interpolate(original_array, target_shape,method='linear'):
    """Resize images using interpolation. For mask-like images, use 'nearest' method, for fast interpolation of regular images use 'linear'

    Arguments:
        original_array -- np.ndarray; image to be resized
        target_shape -- tuple(int,int); shape to resize the image to

    Keyword Arguments:
        method -- str; method for use in scipy.interpolate.RegularGridInterpolator. Valid methods are: ['linear','nearest','slinear','cubic','quintic','pchip'] (default: {'linear'})

    Returns:
        newarr -- np.ndarray; interpolated image with shape target_shape
    """
    ts0, ts1 = target_shape
    ogarr = np.array(original_array)
    os0, os1 = ogarr.shape
    if target_shape == ogarr.shape:
        return ogarr
    methods = ['linear','nearest','slinear','cubic','quintic','pchip']
    if method not in methods:
        raise ValueError(f"{method} is not a valid method. Valid methods are: {methods}.")
    x, y = np.arange(os0), np.arange(os1)
    grid = RegularGridInterpolator((x,y), ogarr,method=method)
    a, b = np.meshgrid(np.linspace(0,int(os0-1),int(ts0)), np.linspace(0,int(os1-1),int(ts1)),indexing='ij')
    points = np.concatenate((a.reshape(int(ts0),int(ts1),1),b.reshape(int(ts0),int(ts1),1)),axis=-1)
    newarr = grid(points)
    return newarr


def extend_background(original_array,target_shape,background_array=None):
    """Extend the background of a trench image so the image has shape target_shape

    Arguments:
        original_array -- np.ndarray; image to extend the background of
        target_shape -- tuple(int,int); shape of the new image

    Keyword Arguments:
        background_array -- np.ndarray; boolean array same shape as original_array, with True for elements which are part of the image background.
                                        If None, assumes background is around the edges of the image (default: {None})

    Returns:
        _description_
    """
    os0, os1 = original_array.shape
    ts0, ts1 = target_shape
    if background_array is None:
        out = np.median(original_array[-1])*np.zeros([ts0,ts1])
    else:
        out = np.mean(original_array[background_array>0]) + np.std(original_array[background_array>0])*np.random.randn(ts0,ts1)/10
    edge = (ts1-os1)//2
    try:
        out[-os0:,edge:-edge] = original_array
    except:
        out[-os0:,edge:-edge-1] = original_array
    return out


def resize_mask(mask, resize_shape, ret_label):
    """
    Resize masks while maintaining their connectivity and values

    :param np.ndarray mask: Input mask
    :param tuple(int, int) resize_shape: Shape to resize the mask to
    :param bool ret_label: Whether to return labeled or bool masks
    :return: Resized mask
    :rtype: np.ndarray
    """
    labeled_mask = label(mask > 0, connectivity=1)
    labeled_mask = resize(labeled_mask, resize_shape, order=0, mode='reflect', cval=0, clip=True, preserve_range=True,
                          anti_aliasing=False, anti_aliasing_sigma=None).astype(int)
    mask_borders = find_boundaries(labeled_mask, mode="thick", connectivity=1)
    labeled_mask = np.where(mask_borders, 0, labeled_mask)
    if ret_label:
        return labeled_mask
    else:
        return labeled_mask > 0


def histogram_intersection(h1, h2, bins):
    sm = 0
    for i in range(bins):
        sm += min(h1[i], h2[i])
    return sm

import warnings
def misc_load_img(dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tifffile.imread(BytesIO(pkgutil.get_data(__name__, dir)))


def get_sample_images():

    """
    Return a dict of sample mother machine images.

    

    Parameters
    ----------

    Returns
    -------
    dict
        A dict with sample images, current keys are: "E. coli 100x", "E. coli 100x stationary", "E. coli DeLTA"
    """

    Ecoli100x = misc_load_img("sample_images/sample_100x.tiff")
    Ecoli100x_stationary = misc_load_img("sample_images/sample_100x_stationary.tiff")
    Ecoli_DeLTA = misc_load_img("sample_images/sample_DeLTA.tiff")
    return {
        "E. coli 100x": Ecoli100x,
        "E. coli 100x stationary": Ecoli100x_stationary,
        "E. coli DeLTA": Ecoli_DeLTA
    }


def unet_weight_map(y, wc=None, w0=10, sigma=5):
    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.

    

    Parameters
    ----------
    mask: Numpy array
        2D array of shape (image_height, image_width) representing binary mask
        of objects.
    wc: dict
        Dictionary of weight classes.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.

    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).



    References
    ----------
    Taken from the original U-net paper [1]_
    
    .. [1] Ronneberger, O., Fischer, P., Brox, T. (2015). 
       U-Net: Convolutional Networks for Biomedical Image Segmentation. 
       In: Navab, N., Hornegger, J., Wells, W., Frangi, A. (eds) 
       Medical Image Computing and Computer-Assisted Intervention – 
       MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science(), 
       vol 9351. Springer, Cham. https://doi.org/10.1007/978-3-319-24574-4_28

    """
 
    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:, :, i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:, :, 0]
        d2 = distances[:, :, 1]
        w = w0 * np.exp(-1 / 2 * ((d1 + d2) / sigma) ** 2) * no_labels
    else:
        w = np.zeros_like(y)
    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w
