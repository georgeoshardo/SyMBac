from skimage.measure import label
import tifffile
import pkgutil
from io import BytesIO

def resize_mask(mask, resize_shape, ret_label):
    """
    Resize masks while maintaining their connectivity and values
    """
    labeled_mask = label(mask>0,connectivity=1)
    labeled_mask = resize(labeled_mask,resize_shape, order=0, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=None).astype(int)
    mask = resize(mask,resize_shape, order=0, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=None).astype(int)
    mask_borders = find_boundaries(labeled_mask,mode="thick", connectivity=1)
    labeled_mask = np.where(mask_borders, 0,labeled_mask)
    if ret_label:
        return labeled_mask
    else:
        return labeled_mask > 0
    
def histogram_intersection(h1, h2,bins):
    sm = 0
    for i in range(bins):
        sm += min(h1[i], h2[i])
    return sm
   
def misc_load_img(dir):
    return tifffile.imread(BytesIO(pkgutil.get_data(__name__, dir)))

def get_sample_images():
    Ecoli100x = misc_load_img("sample_images/sample_100x.tiff")
    Ecoli100x_stationary = misc_load_img("sample_images/sample_100x_stationary.tiff")
    return {
        "E. coli 100x" : Ecoli100x,
        "E. coli 100x stationary" : Ecoli100x_stationary
        }
        
def unet_weight_map(y, wc=None, w0 = 10, sigma = 5):

    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.

    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

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
    """

    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
    else:
        w = np.zeros_like(y)
    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w
