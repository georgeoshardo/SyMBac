from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import warnings
import os
import datetime
from tqdm import tqdm
from zipfile import ZipFile, ZIP_DEFLATED
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
from scipy.ndimage.measurements import find_objects
from scipy.optimize import minimize_scalar
from skimage.measure import regionprops
from csbdeep.utils import _raise
from csbdeep.utils.six import Path

from .matching import matching_dataset
import splinegenerator as sg
import cv2


def gputools_available():
    try:
        import gputools
    except:
        return False
    return True


def path_absolute(path_relative):
    """ Get absolute path to resource"""
    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, path_relative)


def _is_power_of_2(i):
    assert i > 0
    e = np.log2(i)
    return e == int(e)


def _normalize_grid(grid,n):
    try:
        grid = tuple(grid)
        (len(grid) == n and
         all(map(np.isscalar,grid)) and
         all(map(_is_power_of_2,grid))) or _raise(TypeError())
        return tuple(int(g) for g in grid)
    except (TypeError, AssertionError):
        raise ValueError("grid = {grid} must be a list/tuple of length {n} with values that are power of 2".format(grid=grid, n=n))


def _edt_dist_func(anisotropy):
    try:
        from edt import edt as edt_func
        # raise ImportError()
        dist_func = lambda img: edt_func(np.ascontiguousarray(img>0), anisotropy=anisotropy)
    except ImportError:
        dist_func = lambda img: distance_transform_edt(img, sampling=anisotropy)
    return dist_func


def _edt_prob(lbl_img, anisotropy=None):
    constant_img = lbl_img.min() == lbl_img.max() and lbl_img.flat[0] > 0
    if constant_img:
        lbl_img = np.pad(lbl_img, ((1,1),)*lbl_img.ndim, mode='constant')
        warnings.warn("EDT of constant label image is ill-defined. (Assuming background around it.)")
    dist_func = _edt_dist_func(anisotropy)
    prob = np.zeros(lbl_img.shape,np.float32)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img==l
        edt = dist_func(mask)[mask]
        prob[mask] = edt/(np.max(edt)+1e-10)
    if constant_img:
        prob = prob[(slice(1,-1),)*lbl_img.ndim].copy()
    return prob


def edt_prob(lbl_img, anisotropy=None):
    """Perform EDT on each labeled object and normalize."""
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    constant_img = lbl_img.min() == lbl_img.max() and lbl_img.flat[0] > 0
    if constant_img:
        lbl_img = np.pad(lbl_img, ((1,1),)*lbl_img.ndim, mode='constant')
        warnings.warn("EDT of constant label image is ill-defined. (Assuming background around it.)")
    dist_func = _edt_dist_func(anisotropy)
    objects = find_objects(lbl_img)
    prob = np.zeros(lbl_img.shape,np.float32)
    for i,sl in enumerate(objects,1):
        # i: object label id, sl: slices of object in lbl_img
        if sl is None: continue
        interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
        # 1. grow object slice by 1 for all interior object bounding boxes
        # 2. perform (correct) EDT for object with label id i
        # 3. extract EDT for object of original slice and normalize
        # 4. store edt for object only for pixels of given label id i
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl,interior)]==i
        mask = grown_mask[shrink_slice]
        edt = dist_func(grown_mask)[shrink_slice][mask]
        prob[sl][mask] = edt/(np.max(edt)+1e-10)
    if constant_img:
        prob = prob[(slice(1,-1),)*lbl_img.ndim].copy()
    return prob


def _fill_label_holes(lbl_img, **kwargs):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img==l
        mask_filled = binary_fill_holes(mask,**kwargs)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled


def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""
    # TODO: refactor 'fill_label_holes' and 'edt_prob' to share code
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i,sl in enumerate(objects,1):
        if sl is None: continue
        interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl,interior)]==i
        mask_filled = binary_fill_holes(grown_mask,**kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def sample_points(n_samples, mask, prob=None, b=2):
    """sample points to draw some of the associated polygons"""
    if b is not None and b > 0:
        # ignore image boundary, since predictions may not be reliable
        mask_b = np.zeros_like(mask)
        mask_b[b:-b,b:-b] = True
    else:
        mask_b = True

    points = np.nonzero(mask & mask_b)

    if prob is not None:
        # weighted sampling via prob
        w = prob[points[0],points[1]].astype(np.float64)
        w /= np.sum(w)
        ind = np.random.choice(len(points[0]), n_samples, replace=True, p=w)
    else:
        ind = np.random.choice(len(points[0]), n_samples, replace=True)

    points = points[0][ind], points[1][ind]
    points = np.stack(points,axis=-1)
    return points


def calculate_extents(lbl, func=np.median):
    """ Aggregate bounding box sizes of objects in label images. """
    if isinstance(lbl,(tuple,list)) or (isinstance(lbl,np.ndarray) and lbl.ndim==4):
        return func(np.stack([calculate_extents(_lbl,func) for _lbl in lbl], axis=0), axis=0)

    n = lbl.ndim
    n in (2,3) or _raise(ValueError("label image should be 2- or 3-dimensional (or pass a list of these)"))

    regs = regionprops(lbl)
    if len(regs) == 0:
        return np.zeros(n)
    else:
        extents = np.array([np.array(r.bbox[n:])-np.array(r.bbox[:n]) for r in regs])
        return func(extents, axis=0)


def polyroi_bytearray(x,y,pos=None,subpixel=True):
    """ Byte array of polygon roi with provided x and y coordinates
        See https://github.com/imagej/imagej1/blob/master/ij/io/RoiDecoder.java
    """
    import struct
    def _int16(x):
        return int(x).to_bytes(2, byteorder='big', signed=True)
    def _uint16(x):
        return int(x).to_bytes(2, byteorder='big', signed=False)
    def _int32(x):
        return int(x).to_bytes(4, byteorder='big', signed=True)
    def _float(x):
        return struct.pack(">f", x)

    subpixel = bool(subpixel)
    # add offset since pixel center is at (0.5,0.5) in ImageJ
    x_raw = np.asarray(x).ravel() + 0.5
    y_raw = np.asarray(y).ravel() + 0.5
    x = np.round(x_raw)
    y = np.round(y_raw)
    assert len(x) == len(y)
    top, left, bottom, right = y.min(), x.min(), y.max(), x.max() # bbox

    n_coords = len(x)
    bytes_header = 64
    bytes_total = bytes_header + n_coords*2*2 + subpixel*n_coords*2*4
    B = [0] * bytes_total
    B[ 0: 4] = map(ord,'Iout')   # magic start
    B[ 4: 6] = _int16(227)       # version
    B[ 6: 8] = _int16(0)         # roi type (0 = polygon)
    B[ 8:10] = _int16(top)       # bbox top
    B[10:12] = _int16(left)      # bbox left
    B[12:14] = _int16(bottom)    # bbox bottom
    B[14:16] = _int16(right)     # bbox right
    B[16:18] = _uint16(n_coords) # number of coordinates
    if subpixel:
        B[50:52] = _int16(128)   # subpixel resolution (option flag)
    if pos is not None:
        B[56:60] = _int32(pos)   # position (C, Z, or T)

    for i,(_x,_y) in enumerate(zip(x,y)):
        xs = bytes_header + 2*i
        ys = xs + 2*n_coords
        B[xs:xs+2] = _int16(_x - left)
        B[ys:ys+2] = _int16(_y - top)

    if subpixel:
        base1 = bytes_header + n_coords*2*2
        base2 = base1 + n_coords*4
        for i,(_x,_y) in enumerate(zip(x_raw,y_raw)):
            xs = base1 + 4*i
            ys = base2 + 4*i
            B[xs:xs+4] = _float(_x)
            B[ys:ys+4] = _float(_y)

    return bytearray(B)


def export_imagej_rois(fname, polygons, set_position=True, subpixel=True, compression=ZIP_DEFLATED):
    """ polygons assumed to be a list of arrays with shape (id,2,c) """

    if isinstance(polygons,np.ndarray):
        polygons = (polygons,)

    fname = Path(fname)
    if fname.suffix == '.zip':
        fname = fname.with_suffix('')

    with ZipFile(str(fname)+'.zip', mode='w', compression=compression) as roizip:
        for pos,polygroup in enumerate(polygons,start=1):
            for i,poly in enumerate(polygroup,start=1):
                roi = polyroi_bytearray(poly[1],poly[0], pos=(pos if set_position else None), subpixel=subpixel)
                roizip.writestr('{pos:03d}_{i:03d}.roi'.format(pos=pos,i=i), roi)


def optimize_threshold(Y, Yhat, model, nms_thresh, measure='accuracy', iou_threshs=[0.3,0.5,0.7], bracket=None, tol=1e-2, maxiter=20, verbose=1):
    """ Tune prob_thresh for provided (fixed) nms_thresh to maximize matching score (for given measure and averaged over iou_threshs). """
    np.isscalar(nms_thresh) or _raise(ValueError("nms_thresh must be a scalar"))
    iou_threshs = [iou_threshs] if np.isscalar(iou_threshs) else iou_threshs
    values = dict()

    if bracket is None:
        max_prob = max([np.max(prob) for prob, dist in Yhat])
        bracket = max_prob/2, max_prob
    # print("bracket =", bracket)

    with tqdm(total=maxiter, disable=(verbose!=1), desc="NMS threshold = %g" % nms_thresh) as progress:

        def fn(thr):
            prob_thresh = np.clip(thr, *bracket)
            value = values.get(prob_thresh)
            if value is None:
                Y_instances = [model._instances_from_prediction(y.shape, *prob_dist, prob_thresh=prob_thresh, nms_thresh=nms_thresh)[0] for y,prob_dist in zip(Y,Yhat)]
                stats = matching_dataset(Y, Y_instances, thresh=iou_threshs, show_progress=False, parallel=True)
                values[prob_thresh] = value = np.mean([s._asdict()[measure] for s in stats])
            if verbose > 1:
                print("{now}   thresh: {prob_thresh:f}   {measure}: {value:f}".format(
                    now = datetime.datetime.now().strftime('%H:%M:%S'),
                    prob_thresh = prob_thresh,
                    measure = measure,
                    value = value,
                ), flush=True)
            else:
                progress.update()
                progress.set_postfix_str("{prob_thresh:.3f} -> {value:.3f}".format(prob_thresh=prob_thresh, value=value))
                progress.refresh()
            return -value

        opt = minimize_scalar(fn, method='golden', bracket=bracket, tol=tol, options={'maxiter': maxiter})

    verbose > 1 and print('\n',opt, flush=True)
    return opt.x, -opt.fun


def wrapIndex(t, k, M, half_support):
    wrappedT = t - k
    t_left = t - half_support
    t_right = t + half_support
    if k < t_left:
        if t_left <= k + M <= t_right:
            wrappedT = t - (k + M)
    elif k > t + half_support:
        if t_left <= k - M <= t_right:
            wrappedT = t - (k - M)
    return wrappedT


def phi_generator(M, contoursize_max):
    ts = np.linspace(0, float(M), num=contoursize_max, endpoint=False)
    wrapped_indices = np.array([[wrapIndex(t, k, M, 2)
                                 for k in range(M)] for t in ts])
    vfunc = np.vectorize(sg.B3().value)
    phi = vfunc(wrapped_indices)     
    phi = phi.astype(np.float32)
    np.save('phi_' + str(M) + '.npy',phi)
    return
    
    
def grid_generator(M, patch_size, grid_subsampled):
    coord = np.ones((patch_size[0],patch_size[1],M,2))

    xgrid_points = np.linspace(0,coord.shape[0]-1,coord.shape[0])
    ygrid_points = np.linspace(0,coord.shape[1]-1,coord.shape[1])
    xgrid, ygrid = np.meshgrid(xgrid_points,ygrid_points)
    xgrid, ygrid = np.transpose(xgrid), np.transpose(ygrid)
    grid = np.stack((xgrid,ygrid),axis = 2)
    grid = np.expand_dims(grid, axis = 2)
    grid = np.repeat(grid, coord.shape[2], axis = 2)
    grid = np.expand_dims(grid, axis = 0)

    grid = grid[:,0::grid_subsampled[0],0::grid_subsampled[1]]
    grid = grid.astype(np.float32)
    np.save('grid_' + str(M) + '.npy', grid)
    return


def get_contoursize_max(Y_trn):
    contoursize = []
    for i in range(len(Y_trn)):
        mask = Y_trn[i]
        obj_list = np.unique(mask)
        obj_list = obj_list[1:]  
        
        for j in range(len(obj_list)):  
            mask_temp = mask.copy()     
            mask_temp[mask_temp != obj_list[j]] = 0
            mask_temp[mask_temp > 0] = 1
            
            mask_temp = mask_temp.astype(np.uint8)    
            contours,_ = cv2.findContours(mask_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            areas = [cv2.contourArea(cnt) for cnt in contours]    
            max_ind = np.argmax(areas)
            contour = np.squeeze(contours[max_ind])
            contour = np.reshape(contour,(-1,2))
            contour = np.append(contour,contour[0].reshape((-1,2)),axis=0)
            contoursize = np.append(contoursize,contour.shape[0])
            
    contoursize_max = np.amax(contoursize)            
    return contoursize_max


# TODO: clean
# not for evaluating performance in non-star-convex objects
def iou(labelmap_gt, labelmap_pred):
    iou_list = []
    for i in range(len(labelmap_gt)):   
        mask_gt = labelmap_gt[i]
        mask_gt[mask_gt>0] = 1
        
        mask_pred = labelmap_pred[i]
        mask_pred[mask_pred>0] = 1
        
        intersection = np.logical_and(mask_gt, mask_pred)
        union = np.logical_or(mask_gt, mask_pred)
        iou = np.sum(intersection) / np.sum(union)
        
        iou_list.append(iou)        
    return iou_list
        

# TODO: clean
# use for evaluating performance in non-star-convex objects
def iou_objectwise(labelmap_gt, labelmap_pred):
    iou_list = []
    for i in range(len(labelmap_gt)):
        iou_img = []
        mask_gt = labelmap_gt[i]
        
        mask_pred = labelmap_pred[i]
        mask_matched = np.zeros(mask_pred.shape) 
        
        obj_list_gt = np.unique(mask_gt)
        obj_list_gt = obj_list_gt[1:]
        
        obj_list_pred = np.unique(mask_pred)
        obj_list_pred = obj_list_pred[1:]        

        mask_gt_tmp = mask_gt.copy()
        
        for j in range(len(obj_list_pred)):
            mask_pred_obj = mask_pred.copy()
            mask_pred_obj[mask_pred_obj != obj_list_pred[j]] = 0
            mask_pred_obj[mask_pred_obj>0] = 1            
            
            mask_gt_all = mask_gt_tmp.copy()
            mask_gt_all[mask_gt_all>0] = 1
            
            intersection = np.logical_and(mask_gt_all, mask_pred_obj)  
            
            idx_nonzero = np.argwhere(intersection)               
            if(len(idx_nonzero) != 0):            
                idx_nonzero = idx_nonzero[0]
                label = mask_gt_tmp[idx_nonzero[0],idx_nonzero[1]]
                mask_gt_obj = mask_gt_tmp.copy()
                mask_gt_tmp[mask_gt_tmp==label] = 0                
                
                mask_gt_obj[mask_gt_obj != label] = 0
                mask_gt_obj[mask_gt_obj>0] = 1
                
                intersection_obj = np.logical_and(mask_gt_obj, mask_pred_obj)
                union_obj = np.logical_or(mask_gt_obj, mask_pred_obj)
                iou = np.sum(intersection_obj) / np.sum(union_obj)        
                iou_img.append(iou)
            else:
                iou_img.append(0)        
        iou_img = np.asarray(iou_img)
        iou_img_mean = np.mean(iou_img)
        iou_list.append(iou_img_mean)
    return iou_list