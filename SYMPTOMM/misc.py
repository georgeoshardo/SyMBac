from skimage.measure import label

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