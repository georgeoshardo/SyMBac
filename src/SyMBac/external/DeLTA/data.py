'''
This file contains function definitions for data manipulations and input/output
operations.

@author: jblugagne
'''
from __future__ import print_function
import numpy as np 
import os, glob, re, random, warnings, copy, importlib
import skimage.io as io
import tifffile
import skimage.transform as trans
from skimage.measure import label
from skimage.morphology import square, binary_opening, remove_small_objects
from scipy import interpolate

# Try to import elastic deformations, issue warning if not found:
if importlib.util.find_spec("elasticdeform") is None:
    warnings.warn("Could not load elastic deformations module.")
else:
    import elasticdeform
        



#%% UTILITIES:

def binarizerange(i):
    '''
    This function will binarize a numpy array by thresholding it in the middle
    of its range

    Parameters
    ----------
    i : 2D numpy array
        Input array/image.

    Returns
    -------
    newi : 2D numpy array
        Binarized image.

    '''
    newi = i
    newi[i > (np.amin(i)+np.amax(i))/2] = 1
    newi[i <= (np.amin(i)+np.amax(i))/2] = 0
    return newi

def readreshape(filename, target_size = (256,32), binarize = False, order = 1, rangescale = True):
    '''
    Read image from disk and format it

    Parameters
    ----------
    filename : string
        Path to file. Only PNG, JPG or single-page TIFF files accepted
    target_size : tupe of int, optional
        Size to reshape the image. 
        The default is (256,32).
    binarize : bool, optional
        Use the binarizerange() function on the image.
        The default is False.
    order : int, optional
        interpolation order (see skimage.transform.warp doc). 
        0 is nearest neighbor
        1 is bilinear
        The default is 1.
    rangescale : bool, optional
        Scale array image values to 0-1 if True. 
        The default is True.

    Raises
    ------
    ValueError
        Raised if image file is not a PNG, JPEG, or TIFF file.

    Returns
    -------
    i : numpy 2d array of floats
        Loaded array.

    '''
    fext = os.path.splitext(filename)[1].lower()
    if fext in ('.png', '.jpg', '.jpeg'):
        i = io.imread(filename,as_gray = True)
    elif fext in ('.tif', '.tiff'):
        i = tifffile.imread(filename)
    else:
        raise ValueError('Only PNG, JPG or single-page TIF files accepted')
        
    i = trans.resize(i, target_size, anti_aliasing=True, order=order)
    if binarize:
        i = binarizerange(i)
    i = np.reshape(i,i.shape + (1,) )
    if rangescale:
        i = (i-np.min(i))/np.ptp(i)
    return i

def postprocess(images,square_size=5,min_size=None):
    '''
    A generic binary image cleaning function based on mathematical morphology.

    Parameters
    ----------
    images : 2D or 3D numpy array
        Input image or images stacked along axis=0.
    square_size : int, optional
        Size of the square structuring element to use for morphological opening
        The default is 5.
    min_size : int or None, optional
        Remove objects smaller than this minimum area value. If None, the 
        operation is not performed.
        The default is None.

    Returns
    -------
    images : 2D numpy array
        Cleaned, binarized images. Note that the dimensions are squeezed before 
        return (see numpy.squeeze doc)

    '''
    
    # Expand dims if 2D:
    if images.ndim == 2:
        images = np.expand_dims(images,axis=0)
    
    # Generate struturing element:
    selem = square(square_size)
    
    # Go through stack:
    for index, I in enumerate(images):
        I = binarizerange(I)
        I = binary_opening(I,selem=selem)
        if min_size is not None:
            I = remove_small_objects(I,min_size=min_size)
        images[index,:,:] = I
        
    return np.squeeze(images)
        
# def weightmap(mask, classweights, params):
#     '''
#     This function computes the weight map as described in the original U-Net
#     paper to force the model to learn borders 
#     (Not used, too slow. We preprocess the weight maps in matlab)
#     '''

#     lblimg,lblnb = label(mask[:,:,0],connectivity=1,return_num=True)
#     distance_array = float('inf')*np.ones(mask.shape[0:2] + (max(lblnb,2),))
#     for i in range(0,lblnb):
#         [_,distance_array[:,:,i]] = medial_axis((lblimg==i+1)==0,return_distance=True)
#     distance_array = np.sort(distance_array,axis=-1)
#     weightmap = params["w0"]*np.exp(-np.square(np.sum(distance_array[:,:,0:2],-1))/(2*params["sigma"]^2))
#     weightmap = np.multiply(weightmap,mask[:,:,0]==0)
#     weightmap = np.add(weightmap,(mask[:,:,0]==0)*classweights[0])
#     weightmap = np.add(weightmap,(mask[:,:,0]==1)*classweights[1])
#     weightmap = np.reshape(weightmap, weightmap.shape + (1,))
    
#     return weightmap



#%% DATA AUGMENTATION
def data_augmentation(images_input, aug_par, order=0):
    '''
    Data augmentation function

    Parameters
    ----------
    images_input : list of 2D numpy arrays of floats
        Images to apply augmentation operations to.
    aug_par : dict
        Augmentation operations parameters. Accepted key-value pairs:
            illumination_voodoo: bool. Whether to apply the illumination 
                voodoo operation.
            histogram_voodoo: bool. Whether to apply the histogram voodoo 
                operation.
            elastic_deformation: dict. If key exists, the elastic deformation
                operation is applied. The parameters are given as key-value 
                pairs. sigma values are given under the sigma key, deformation
                points are given under the points key. See elasticdeform doc.
            gaussian_noise: float. Apply gaussian noise to the image. The 
                sigma value of the gaussian noise is uniformly sampled between
                0 and +gaussian_noise.
            horizontal_flip: bool. Whether to flip the images horizontally. 
                Input images have a 50% chance of being flipped
            vertical_flip: bool. Whether to flip the images vertically. 
                Input images have a 50% chance of being flipped
            rotations90d: bool. Whether to randomly rotate the images in 90°
                increments. Each 90° rotation has a 25% chance of happening
            rotation: int/float. Range of random rotation to apply. The angle 
                is uniformly sampled in the range [-rotation, +rotation]
            zoom: float. Range of random "zoom" to apply. The image
                is randomly zoomed by a factor that is sampled from an 
                exponential distribution with a lamba of 3/zoom. The random
                factor is clipped at +zoom.
            shiftX: int/float. The range of random shifts to apply along X. A
                uniformly sampled shift between [-shiftX, +shiftX] is applied
            shiftY: int/float. The range of random shifts to apply along Y. A
                uniformly sampled shift between [-shiftY, +shiftY] is applied
            Note that the same operations are applied to all inputs.
    order : int or list/tuple of ints, optional
        Interpolation order to use for each image in the input stack. If order
        is a scalar, the same order is applied to all images. If a list of
        orders is provided, each image in the stack will have its own operaiton
        order. See skimage.transform.wrap doc.
        Note that the histogram voodoo operation is only applied to images with
        a non-zero order.
        The default is 0.

    Returns
    -------
    output : list of 2D numpy arrays of floats
        Augmented images array.

    '''
    
    #processing inputs / initializing variables::
    output = list(images_input)
    if np.isscalar(order):
        orderlist = [order] * len(images_input)
    else:
        orderlist = list(order)
    
    # Apply augmentation operations:
    
    if "illumination_voodoo" in aug_par:
        if aug_par["illumination_voodoo"]:
            for index, item in enumerate(output):
                if order[index] > 0: # Not super elegant, but tells me if binary or grayscale image
                    output[index] = illumination_voodoo(item)
                    
    #TODO: Illumination voodoo 2D
                    
    if "histogram_voodoo" in aug_par:
        if aug_par["histogram_voodoo"]:
            for index, item in enumerate(output):
                if order[index] > 0: # Not super elegant, but tells me if binary or grayscale image
                    output[index] = histogram_voodoo(item)
                    
    
    if 'gaussian_noise' in aug_par:
        if aug_par['gaussian_noise']:
            sigma = np.random.rand()*aug_par['gaussian_noise']
            for index, item in enumerate(output):
                if order[index] > 0: # Not super elegant, but tells me if binary or grayscale image
                    item = item + np.random.normal(0,sigma,item.shape) # Add Gaussian noise
                    output[index] = (item-np.min(item))/np.ptp(item) # Rescale to 0-1
                
    if "elastic_deformation" in aug_par:
        output = elasticdeform.deform_random_grid(output,
                                                  sigma=aug_par["elastic_deformation"]["sigma"],
                                                  points=aug_par["elastic_deformation"]["points"],
                                                  order=[i*3 for i in orderlist], # Using bicubic interpolation instead of bilinear here
                                                  mode='nearest',
                                                  axis=(0,1),
                                                  prefilter=False)
    
    if "horizontal_flip" in aug_par:
        if aug_par["horizontal_flip"]:
            if random.randint(0,1): #coin flip
                for index, item in enumerate(output):
                    output[index] = np.fliplr(item)
                    
    if "vertical_flip" in aug_par:
        if aug_par["vertical_flip"]:
            if random.randint(0,1): #coin flip
                for index, item in enumerate(output):
                    output[index] = np.flipud(item)
                    
    if "rotations_90d" in aug_par: # Only works with square images right now!
        if aug_par["rotations_90d"]:
            rot = random.randint(0,3)*90
            if rot>0: 
                for index, item in enumerate(output):
                    output[index] = trans.rotate(item,rot,mode='edge',order=orderlist[index])
    
    if "rotation" in aug_par:
        rot = random.uniform(-aug_par["rotation"],aug_par["rotation"])
        for index, item in enumerate(output):
            output[index] = trans.rotate(item,rot,mode='edge',order=orderlist[index])
    
    if "zoom" in aug_par:
        zoom = random.expovariate(3*1/aug_par["zoom"]) # I want most of them to not be too zoomed
        zoom = aug_par["zoom"] if zoom > aug_par["zoom"] else zoom
    else:
        zoom = 0
        
    if "shiftX" in aug_par:
        shiftX = random.uniform(-aug_par["shiftX"],aug_par["shiftX"])
    else:
        shiftX = 0
        
    if "shiftY" in aug_par:
        shiftY = random.uniform(-aug_par["shiftY"],aug_par["shiftY"])
    else:
        shiftY = 0
        
    if any(abs(x)>0 for x in [zoom,shiftX,shiftY]):
        for index, item in enumerate(output):
            output[index] = zoomshift(item,zoom+1,shiftX,shiftY, order=orderlist[index])
    
    return output

def zoomshift(I,zoomlevel,shiftX,shiftY, order=0):
    '''
    This function zooms and shifts images.

    Parameters
    ----------
    I : 2D numpy array
        input image.
    zoomlevel : float
        Additional zoom to apply to the image.
    shiftX : int
        X-axis shift to apply to the image, in pixels.
    shiftY : int
        Y-axis shift to apply to the image, in pixels.
    order : int, optional
        Interpolation order. The default is 0.

    Returns
    -------
    I : 2D numpy array
        Zoomed and shifted image of same size as input.

    '''
    
    oldshape = I.shape
    I = trans.rescale(I,zoomlevel,mode='edge',multichannel=False, order=order)
    shiftX = shiftX * I.shape[0]
    shiftY = shiftY * I.shape[1]
    I = shift(I,(shiftY, shiftX),order=order) # For some reason it looks like X & Y are inverted?
    i0 = (round(I.shape[0]/2 - oldshape[0]/2), round(I.shape[1]/2 - oldshape[1]/2))
    I = I[i0[0]:(i0[0]+oldshape[0]), i0[1]:(i0[1]+oldshape[1])]
    return I
    
def shift(image, vector, order=0):
    '''
    Image shifting function

    Parameters
    ----------
    image : 2D numpy array
        Input image.
    vector : tuple of ints
        Translation/shit vector.
    order : int, optional
        Interpolation order. The default is 0.

    Returns
    -------
    shifted : 2D numpy image
        Shifted image.

    '''
    transform = trans.AffineTransform(translation=vector)
    shifted = trans.warp(image, transform, mode='edge',order=order)

    return shifted

def histogram_voodoo(image,num_control_points=3):
    '''
    This function kindly provided by Daniel Eaton from the Paulsson lab.
    It performs an elastic deformation on the image histogram to simulate
    changes in illumination

    Parameters
    ----------
    image : 2D numpy array
        Input image.
    num_control_points : int, optional
        Number of inflection points to use on the histogram conversion curve. 
        The default is 3.

    Returns
    -------
    2D numpy array
        Modified image.

    '''
    control_points = np.linspace(0,1,num=num_control_points+2)
    sorted_points = copy.copy(control_points)
    random_points = np.random.uniform(low=0.1,high=0.9,size=num_control_points)
    sorted_points[1:-1] = np.sort(random_points)
    mapping = interpolate.PchipInterpolator(control_points, sorted_points)
    
    return mapping(image)

def illumination_voodoo(image,num_control_points=5):
    '''
    This function inspired by the one above.
    It simulates a variation in illumination along the length of the chamber

    Parameters
    ----------
    image : 2D numpy array
        Input image.
    num_control_points : int, optional
        Number of inflection points to use on the illumination multiplication
        curve. 
        The default is 5.

    Returns
    -------
    newimage : 2D numpy array
        Modified image.

    '''
    
    # Create a random curve along the length of the chamber:
    control_points = np.linspace(0,image.shape[0]-1,num=num_control_points)
    random_points = np.random.uniform(low=0.1,high=0.9,size=num_control_points)
    mapping = interpolate.PchipInterpolator(control_points, random_points)
    curve = mapping(np.linspace(0,image.shape[0]-1,image.shape[0]))
    # Apply this curve to the image intensity along the length of the chamebr:
    newimage = np.multiply(image,
                           np.reshape(
                                   np.tile(
                                           np.reshape(curve,curve.shape + (1,)), (1, image.shape[1])
                                           )
                                   ,image.shape
                                   )
                           )
    # Rescale values to original range:
    newimage = np.interp(newimage, (newimage.min(), newimage.max()), (image.min(), image.max()))
    
    return newimage



#%% SEGMENTATION FUNCTIONS:

def trainGenerator_seg(batch_size,
                    img_path, mask_path, weight_path,
                    target_size = (256,32),
                    augment_params = {},
                    preload = False,
                    seed = 1):
    '''
    Generator for training the segmentation U-Net.

    Parameters
    ----------
    batch_size : int
        Batch size, number of training samples to concatenate together.
    img_path : string
        Path to folder containing training input images.
    mask_path : string
        Path to folder containing training segmentation groundtruth.
    weight_path : string
        Path to folder containing weight map images.
    target_size : tuple of 2 ints, optional
        Input and output image size. 
        The default is (256,32).
    augment_params : dict, optional
        Data augmentation parameters. See data_augmentation() doc for more info
        The default is {}.
    preload : bool, optional
        Flag to load all training inputs in memory during intialization of the
        generator.
        The default is False.
    seed : int, optional
        Seed for numpy's random generator. see numpy.random.seed() doc
        The default is 1.

    Yields
    ------
    image_arr : 4D numpy array of floats
        Input images for the U-Net training routine. Dimensions of the tensor 
        are (batch_size, target_size[0], target_size[1], 1)
    mask_wei_arr : 4D numpy array of floats
        Output masks and weight maps for the U-Net training routine. Dimensions
        of the tensor are (batch_size, target_size[0], target_size[1], 2)

    '''
    
    preload_mask = []
    preload_img = []
    preload_weight = []
    
    # Get training image files list:
    image_name_arr =    glob.glob(os.path.join(img_path,"*.png")) +\
                        glob.glob(os.path.join(img_path,"*.tif")) +\
			glob.glob(os.path.join(img_path,"*.tiff"))
    
    # If preloading, load the images and compute weight maps:
    if preload:
        for filename in image_name_arr:
            preload_img.append(readreshape(filename, target_size = target_size, order = 1))
            preload_mask.append(readreshape(os.path.join(mask_path,os.path.basename(filename)), target_size = target_size, binarize = True, order = 0, rangescale = False))
            if weight_path is not None:
                preload_weight.append(readreshape(os.path.join(weight_path,os.path.basename(filename)), target_size = target_size, order = 0), rangescale = False)
    
    # Reset the pseudo-random generator:
    random.seed(a=seed)
    
    while True:
        # Reset image arrays:
        image_arr = []
        mask_arr = []
        weight_arr = []
        for _ in range(batch_size):
            # Pick random image index:
            index = random.randrange(0,len(image_name_arr))
            
            if preload:
                # Get from preloaded arrays:
                img = preload_img[index]
                mask = preload_mask[index]
                weight = preload_weight[index]
            else:
                # Read images:
                filename = image_name_arr[index]
                img = readreshape(filename, target_size = target_size, order = 1)
                mask = readreshape(os.path.join(mask_path,os.path.basename(filename)), target_size = target_size, binarize = True, order = 0, rangescale = False)
                if weight_path is not None:
                    weight = readreshape(os.path.join(weight_path,os.path.basename(filename)), target_size = target_size, order = 0, rangescale = False)
                else:
                    weight = []
                    

            # Data augmentation:
            if weight_path is not None:
                [img, mask, weight] = data_augmentation([img, mask, weight], augment_params, order=[1,0,0])
            else:
                [img, mask] = data_augmentation([img, mask], augment_params, order=[1,0])
            
            # Append to output list:
            image_arr.append(img)
            mask_arr.append(mask)
            weight_arr.append(weight)

        # Concatenate masks and weights: (gets unstacked by the loss function)
        image_arr = np.array(image_arr)
        if weight_path is not None:
            mask_wei_arr = np.concatenate((mask_arr,weight_arr),axis=-1)
        else:
            mask_wei_arr = np.array(mask_arr)
        yield (image_arr, mask_wei_arr)


def saveResult_seg(save_path,npyfile, files_list = [], multipage=False):
    '''
    Saves an array of segmentation output images to disk

    Parameters
    ----------
    save_path : string
        Path to save folder.
    npyfile : 3D or 4D numpy array
        Array of segmentation outputs to save to individual files. If 4D, only 
        the images from the first index of axis=3 will be saved.
    files_list : list of strings, optional
        Filenames to save the segmentation masks as. png, tif or jpg extensions
        work.
        The default is [].
    multipage : bool, optional
        Flag to save all output masks as a single, multi-page TIFF file. Note
        that if the file already exists, the masks will be appended to it.
        The default is False.

    Returns
    -------
    None.

    '''
    
    for i,item in enumerate(npyfile):
        if item.ndim == 3:
            img = item[:,:,0]
        else:
            img = item
        if multipage:
            filename = os.path.join(save_path,files_list[0])
            io.imsave(filename,(img*255).astype(np.uint8),plugin='tifffile',append=True)
        else:
            if files_list:
                filename = os.path.join(save_path,files_list[i])
            else:
                filename = os.path.join(save_path,"%d_predict.png"%i)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(filename,(img*255).astype(np.uint8))


def predictGenerator_seg(files_path, files_list = [], target_size = (256,32)):
    '''
    Get a generator for predicting segmentation on new image files 
    once the segmentation U-Net has been trained.

    Parameters
    ----------
    files_path : string
        Path to image files folder.
    files_list : list/tuple of strings, optional
        List of file names to read in the folder. If empty, all 
        files in the folder will be read.
        The default is [].
    target_size : tuple of 2 ints, optional
        Size for the images to be resized.
        The default is (256,32).

    Returns
    ------
    mygen : generator
        Generator that will yield single image files as 4D numpy arrays of
        size (1, target_size[0], target_size[1], 1).

    '''
    
    
    if not files_list:
        files_list = os.listdir(files_path)

        
    def generator(files_path, files_list, target_size):
        for index, fname in enumerate(files_list):
            img = readreshape(os.path.join(files_path,fname),
                              target_size=target_size,
                              order=1)
            img = np.reshape(img,(1,)+img.shape) # Tensorflow needs one extra single dimension (so that it is a 4D tensor)
            yield img
                
    mygen = generator(files_path, files_list, target_size)
    return mygen
    

        
        
        
#%% TRACKING FUNCTIONS
    
def trainGenerator_track(batch_size,
                    img_path, seg_path, previmg_path, segall_path,
                    mother_path, daughter_path,
                    augment_params = {},
                    target_size = (256,32),
                    seed = 1):
    '''
    Generator for training the tracking U-Net.

    Parameters
    ----------
    batch_size : int
        Batch size, number of training samples to concatenate together.
    img_path : string
        Path to folder containing training input images (current timepoint).
    seg_path : string
        Path to folder containing training 'seed' images, ie mask of 1 cell 
        in the previous image to track in the current image.
    previmg_path : string
        Path to folder containing training input images (previous timepoint).
    segall_path : string
        Path to folder containing training 'segall' images, ie mask of all 
        cells in the current image.
    mother_path : string
        Path to folder containing 'mother' tracking groundtruth, ie mask of 
        the seed cell tracked in the current frame. If a division happens
        between frames, the top cell is considered the 'mother' cell
    daughter_path : string
        Path to folder containing 'daughter' tracking groundtruth, ie mask of 
        the daughter of the seed cell in the current frame, if a division has 
        happened.
    augment_params : dict, optional
        Data augmentation parameters. See data_augmentation() doc for more info
        The default is {}.
    target_size : tuple of 2 ints, optional
        Input and output image size. 
        The default is (256,32).
    seed : int, optional
        Seed for numpy's random generator. see numpy.random.seed() doc
        The default is 1.

    Yields
    ------
    inputs_arr : 4D numpy array of floats
        Input images and masks for the U-Net training routine. Dimensions of 
        the tensor are (batch_size, target_size[0], target_size[1], 4)
    outputs_arr : 4D numpy array of floats
        Output masks for the U-Net training routine. Dimensions of the tensor 
        are (batch_size, target_size[0], target_size[1], 3). The third index
        of axis=3 contains 'background' masks, ie the part of the tracking 
        output groundtruth that is not part of the mother or daughter masks
    '''
    
    
    # Initialize variables and arrays:
    image_name_arr =    glob.glob(os.path.join(img_path,"*.png")) +\
                        glob.glob(os.path.join(img_path,"*.tif"))
    
    # Reset the pseudo-random generator:
    random.seed(a=seed)
    
    while True:
        # Reset image arrays:
        img_arr = []
        seg_arr = []
        previmg_arr = []
        segall_arr = []
        mother_arr = []
        daughter_arr = []
        bkgd_arr = []

        for _ in range(batch_size):
            # Pick random image file name:
            index = random.randrange(0,len(image_name_arr))
            filename = image_name_arr[index]
            
            # Read images:
            img = readreshape(filename, target_size = target_size, order = 1)
            seg = readreshape(os.path.join(seg_path,os.path.basename(filename)), target_size = target_size, binarize = True, order = 0, rangescale = False)
            previmg = readreshape(os.path.join(previmg_path,os.path.basename(filename)), target_size = target_size, order = 1)
            segall = readreshape(os.path.join(segall_path,os.path.basename(filename)), target_size = target_size, binarize = True, order = 0, rangescale = False)

            mother = readreshape(os.path.join(mother_path,os.path.basename(filename)), target_size = target_size, binarize = True, order = 0, rangescale = False)
            daughter = readreshape(os.path.join(daughter_path,os.path.basename(filename)), target_size = target_size, binarize = True, order = 0, rangescale = False)
            
            # Create the "background" image: (necessary for categorical cross-entropy)
            bkgd = 1 - (mother + daughter)
            
            # Data augmentation:
            [img, seg, previmg, segall, mother, daughter, bkgd] = data_augmentation([img, seg, previmg, segall, mother, daughter, bkgd],
                                augment_params, order=[1,0,1,0,0,0,0])
            
            # Append to arrays:
            seg_arr.append(seg)
            img_arr.append(img)
            previmg_arr.append(previmg)
            segall_arr.append(segall)
            mother_arr.append(mother)
            daughter_arr.append(daughter)
            bkgd_arr.append(bkgd)
            
        # Concatenare and yield inputs and output:
        inputs_arr = np.concatenate((img_arr,seg_arr,previmg_arr,segall_arr),axis=-1)
        outputs_arr = np.concatenate((mother_arr,daughter_arr,bkgd_arr),axis=-1)
        yield (inputs_arr, outputs_arr)
            
            
# def predictGenerator_track(test_path,files_list = [],target_size = (256,32)):
#     '''
#     This function is a generator for predicting tracking on new image files 
#     once the tracking U-Net has been trained.
    
#     Images are loaded from subfolders in the test_path folder (following the 
#     same nomenclature as in trainGenerator_track())
#     If a list of filenames files_list is specified, it will read only those 
#     images, otherwise it will go through the entire folder.
    
#     Images are resized to target_size upon loading.
#     '''
    
#     if not files_list:
#         files_list = os.listdir(test_path)
#     for index, item in enumerate(files_list):
#             img = readreshape(os.path.join(test_path + 'img/',os.path.basename(item)), target_size = target_size, order = 1)
#             seg = readreshape(os.path.join(test_path + 'seg/',os.path.basename(item)), target_size = target_size, binarize = True, order = 0, rangescale = False)
#             previmg = readreshape(os.path.join(test_path + 'previmg/',os.path.basename(item)), target_size = target_size, order = 1)
#             segall = readreshape(os.path.join(test_path + 'segall/',os.path.basename(item)), target_size = target_size, binarize = True, order = 0, rangescale = False)
            
#             inputs_arr = np.concatenate((img,seg,previmg,segall),axis=-1)
#             inputs_arr = np.reshape(inputs_arr,(1,)+inputs_arr.shape) # Tensorflow needs one extra single dimension (so that it is a 4D tensor)
            
#             yield inputs_arr


def predictCompilefromseg_track(img_path,seg_path, files_list = [],
                                  target_size = (256,32)):
    '''
    Compile an inputs array for tracking prediction with the tracking U-Net, 
    directly from U-Net segmentation masks saved to disk.

    Parameters
    ----------
    img_path : string
        Path to original single-chamber images folder. The filenames are 
        expected in the printf format Position%02d_Chamber%02d_Frame%03d.png
    seg_path : string
        Path to segmentation output masks folder. The filenames must be the 
        same as in the img_path folder.
    files_list : tuple/list of strings, optional
        List of filenames to compile in the img_path and seg_path folders. 
        If empty, all files in the folder will be read.
        The default is [].
    target_size : tuple of 2 ints, optional
        Input and output image size. 
        The default is (256,32).

    Returns
    -------
    inputs_arr : 4D numpy array of floats
        Input images and masks for the tracking U-Net training routine. 
        Dimensions of the tensor are (cells_to_track, target_size[0], 
        target_size[1], 4), with cells_to_track the number of segmented cells
        in all segmentation masks of the files_list.
    seg_name_list : list of strings
        Filenames to save the tracking outputs as. The printf format is 
        Position%02d_Chamber%02d_Frame%03d_Cell%02d.png, with the '_Cell%02d'
        string appended to signal which cell is being seeded/tracked (from top
        to bottom)
        

    '''
    
    img_arr = []
    seg_arr = []
    previmg_arr = []
    segall_arr = []
    seg_name_list = []
    
    if not files_list:
        files_list = os.listdir(img_path)
        
    for index, item in enumerate(files_list):
            filename = os.path.basename(item)
            # Get position, chamber & frame numbers:
            (pos, cha, fra) = list(map(int, re.findall("\d+",filename)))
            if fra > 1:
                prevframename = 'Position' + str(pos).zfill(2) + \
                                '_Chamber' + str(cha).zfill(2) + \
                                '_Frame' + str(fra-1).zfill(3) + '.png'
                img = readreshape(os.path.join(img_path,filename), target_size = target_size, order = 1)
                segall = readreshape(os.path.join(seg_path,filename), target_size = target_size, order = 0, binarize = True, rangescale = False)
                previmg = readreshape(os.path.join(img_path,prevframename), target_size = target_size, order = 1)
                prevsegall = readreshape(os.path.join(seg_path,prevframename), target_size = target_size, order = 0, binarize = True, rangescale = False)
                
                lblimg,lblnb = label(prevsegall[:,:,0],connectivity=1,return_num=True)
                
                for lbl in range(1,lblnb+1):
                    segfilename = 'Position' + str(pos).zfill(2) + \
                                '_Chamber' + str(cha).zfill(2) + \
                                '_Frame' + str(fra).zfill(3) + \
                                '_Cell' + str(lbl).zfill(2) + '.png'

                    seg = lblimg == lbl
                    seg = np.reshape(seg, seg.shape + (1,))
                    seg.astype(segall.dtype) # Output is boolean otherwise

                    seg_arr.append(seg)
                    img_arr.append(img)
                    previmg_arr.append(previmg)
                    segall_arr.append(segall)
                    seg_name_list.append(segfilename)
    
    inputs_arr = np.concatenate((img_arr,seg_arr,previmg_arr,segall_arr),axis=-1)
    return inputs_arr, seg_name_list


def estimateClassweights(gene, num_samples = 30):
    '''
    Estimate the class weights to use with the weighted 
    categorical cross-entropy based on the output of the trainGenerator_track
    output.

    Parameters
    ----------
    gene : generator
        Tracking U-Net training generator. (output of trainGenerator_track)
    num_samples : int, optional
        Number of batches to use for estimation. The default is 30.

    Returns
    -------
    class_weights : tuple of floats
        Relative weights of each class. Note that, if 0 elements of a certain 
        class are present in the samples, the weight for this class will be set 
        to 0.

    '''
    
    sample = next(gene)
    class_counts = [0] * sample[1].shape[-1] # List of 0s
    
    # Run through samples and classes/categories:
    for _ in range(num_samples):
        for i in range(sample[1].shape[-1]):
            class_counts[i] += np.mean(sample[1][..., i])
        sample = next(gene)
        
    # Warning! If 0 elements of a certain class are present in the samples, the
    # weight for this class will be set to 0. This is for the tracking case
    # (Where there are only empty daughter images in the training set)
    # Try changing the num_samples value if this is a problem
    class_weights = [(x/num_samples)**-1 if x!= 0 else 0 for x in class_counts] # Normalize by nb of samples and invert to get weigths, unless x == 0 to avoid Infinite weights or errors
    
    return class_weights


def saveResult_track(save_path,npyfile, files_list = None):
    '''
    Save tracking output masks to disk

    Parameters
    ----------
    save_path : string
        Folder to save images to.
    npyfile : 4D numpy array
        Array of tracking outputs to save to individual files.
    files_list : tuple/list of strings, optional
        Filenames to save the masks as. Note that the 'mother_' and 'daughter_'
        prefixes will be added to those names. If None, numbers will be used.
        The default is None.

    Returns
    -------
    None.

    '''
    
    mothers = npyfile[:,:,:,0]
    daughters = npyfile[:,:,:,1]
    for i,mother in enumerate(mothers):
        if files_list:
            filenameMo = os.path.join(save_path,'mother_' + files_list[i])
        else:
            filenameMo = os.path.join(save_path,"%mother_09d.png"%i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(filenameMo,(mother*255).astype(np.uint8))
    for i,daughter in enumerate(daughters):
        if files_list:
            filenameDa = os.path.join(save_path,'daughter_' + files_list[i])
        else:
            filenameDa = os.path.join(save_path,"%daughter_09d.png"%i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(filenameDa,(daughter*255).astype(np.uint8))
