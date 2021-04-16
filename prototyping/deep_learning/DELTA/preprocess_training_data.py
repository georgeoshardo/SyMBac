import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
from PIL import Image
from numpy import asarray
import os
from skimage.exposure import rescale_intensity
import skimage

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

base_directory = "/home/georgeos/Storage/Dropbox (Cambridge University)/PhD_Georgeos_Hardo/ML_based_segmentation_results/40x_Ph2_test_1.5/training_data/"

training_img_dir = base_directory + "convolutions/"
training_img_files = os.listdir(training_img_dir)

masks_dir = base_directory + "masks/"
masks_files = os.listdir(masks_dir)
weightmap_dir = base_directory + "WEIGHTMAPS"

try: 
    os.mkdir(base_directory + "preprocessed")
    os.mkdir(base_directory + "preprocessed/CROPPED_FILTERED")
    os.mkdir(base_directory + "preprocessed/CROPPED_MASKS")
    os.mkdir(base_directory + "preprocessed/WEIGHTMAPS")
except:
    pass
    
def crop_and_resize(image_dir, LEFT_CROP, RIGHT_CROP):
    image = Image.open(image_dir)
    image = asarray(image)[0:-20,LEFT_CROP:RIGHT_CROP]
    HEIGHT = image.shape[0]
    WIDTH = image.shape[1]
    NEW_HEIGHT = int(np.ceil(HEIGHT*3/64) * 64)
    NEW_WIDTH = int(2 ** np.ceil(np.log2(WIDTH*3)))
    image = rescale_intensity(image)
    image = skimage.img_as_ubyte(image)
    image = Image.fromarray(image)
    image = image.resize((NEW_WIDTH,NEW_HEIGHT),resample=0)
    return image
    
  

LEFT_CROP = 6
RIGHT_CROP = LEFT_CROP + 16

for x in range(len(training_img_files)):
    cropped = crop_and_resize(training_img_dir + training_img_files[x], LEFT_CROP, RIGHT_CROP)
    cropped.save(base_directory + "preprocessed/CROPPED_FILTERED/"+training_img_files[x])
    
for x in range(len(masks_files)):
    cropped = crop_and_resize(masks_dir + masks_files[x], LEFT_CROP, RIGHT_CROP)
    cropped = np.array(cropped)/255
    cropped = skimage.img_as_ubyte(cropped)/255
    cropped = Image.fromarray(cropped)
    cropped.save(base_directory + "preprocessed/CROPPED_MASKS/"+masks_files[x])
    
    
wc = {
    0: 1, # background
    1: 5  # objects
}

for x in range(len(masks_files)):
    image = Image.open(base_directory + "preprocessed/CROPPED_MASKS/" + masks_files[x])
    image = np.array(image)
    w = unet_weight_map(image, wc)
    image = Image.fromarray(w.astype(np.uint8))
    image.save(base_directory + "preprocessed/WEIGHTMAPS/"+masks_files[x])
