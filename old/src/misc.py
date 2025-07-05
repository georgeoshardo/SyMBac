import pkgutil
from io import BytesIO

import numpy as np
#import tifffile
#from scipy.ndimage import distance_transform_edt
#from skimage.measure import label
#from skimage.segmentation import find_boundaries
#from skimage.transform import resize

import math
import pymunk
import typing
if typing.TYPE_CHECKING:
    from symbac.simulation.segments import CellSegment

def calculate_overlap_fraction(segment_a: 'CellSegment', segment_b: 'CellSegment') -> float:
    """
    Calculates the overlapping area of two circles as a fraction of one circle's area.

    Args:
        segment_a: The first cell segment.
        segment_b: The second cell segment.

    Returns:
        The fraction of area that is overlapping (from 0.0 to 1.0).
    """
    # Both segments should have the same radius
    R = segment_a.radius

    # Get the distance between the centers of the two shapes
    pos_a = segment_a.body.position
    pos_b = segment_b.body.position
    d = pos_a.get_distance(pos_b)

    # If the distance is greater than or equal to the sum of radii, there is no overlap.
    if d >= 2 * R:
        return 0.0

    # If the distance is zero, one circle is completely on top of the other.
    if d == 0:
        return 1.0

    # Apply the formula for the area of intersection
    # Breaking it down for clarity
    part1 = 2 * R ** 2 * math.acos(d / (2 * R))
    part2 = (d / 2) * math.sqrt(4 * R ** 2 - d ** 2)
    intersection_area = part1 - part2

    # Calculate the area of a single circle
    circle_area = math.pi * R ** 2

    # Return the fraction
    return intersection_area / circle_area

def generate_color(group_id: int) -> tuple[int, int, int]:
    """
    Generate a unique color based on group_id using HSV color space
    for better visual distinction between cells.
    """
    import colorsys

    # Use golden ratio for better color distribution
    golden_ratio = 0.618033988749895
    hue = (group_id * golden_ratio) % 1.0
    saturation = 0.7 + (group_id % 3) * 0.1  # Vary saturation slightly
    value = 0.8 + (group_id % 2) * 0.2  # Vary brightness slightly

    rgb: tuple[float, float, float] = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)


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
    labeled_mask = resize(
        labeled_mask,
        resize_shape,
        order=0,
        mode="reflect",
        cval=0,
        clip=True,
        preserve_range=True,
        anti_aliasing=False,
        anti_aliasing_sigma=None,
    ).astype(int)
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
        "E. coli DeLTA": Ecoli_DeLTA,
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
