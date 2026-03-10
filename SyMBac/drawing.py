import random

import numpy as np
import noise
from matplotlib import pyplot as plt
from numba import njit
from skimage.measure import label
from skimage.transform import rescale
from skimage.morphology import opening, remove_small_objects
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries
from PIL import Image

div_odd = lambda n: (n // 2, n // 2 + 1)
perc_diff = lambda a, b: (a - b) / b * 100


def apply_cell_texture(OPL_cell, cell_id, scale=70.0, octaves=3, persistence=0.5, lacunarity=1.0, strength=0.5):
    """
    Apply temporally coherent Perlin noise texture to a rasterized OPL cell.

    The noise is anchored to the cell center using coordinates relative to
    (length/2, width/2), so as a cell grows the existing texture remains
    stable and new texture is revealed at the poles.

    Parameters
    ----------
    OPL_cell : np.ndarray
        2D array of optical path length values for one cell.
    cell_id : int
        Unique cell ID, used to seed the Perlin noise base for per-cell consistency.
    scale : float
        Spatial scale of the noise pattern.
    octaves : int
        Number of Perlin noise octaves.
    persistence : float
        Amplitude scaling per octave.
    lacunarity : float
        Frequency scaling per octave.
    strength : float
        Modulation strength (0 = no texture, 1 = full modulation).

    Returns
    -------
    np.ndarray
        The textured OPL cell array.
    """
    length, width = OPL_cell.shape
    i_idx = np.arange(length) - length / 2
    j_idx = np.arange(width) - width / 2
    world_i, world_j = np.meshgrid(i_idx, j_idx, indexing="ij")

    texture = np.vectorize(noise.pnoise2)(
        world_i / 2 / scale,
        world_j / 2 / scale,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        base=int(cell_id) % 1024,
    )
    return OPL_cell * (1 + texture * strength)


def get_crop_bounds_2D(img, tol=0):
    mask = img>tol
    x_idx = np.ix_(mask.any(1),mask.any(0))
    start_row, stop_row, start_col, stop_col = x_idx[0][0][0], x_idx[0][-1][0], x_idx[1][0][0], x_idx[1][0][-1]

    return (start_row, stop_row), (start_col, stop_col)

def crop_image(img, rows, cols, pad):
    (start_row, stop_row) = rows
    (start_col, stop_col) = cols
    if len(img.shape)==3:
        return np.pad(img[:,start_row:stop_row, start_col:stop_col], ((0,0),(pad,pad),(pad,pad)))
    else:
        return np.pad(img[start_row:stop_row, start_col:stop_col], pad)


def raster_cell(length, width, separation, pinching=True, FL = False):
    """
    Produces a rasterised image of a cell with the intensiity of each pixel corresponding to the optical path length
    (thickness) of the cell at that point.

    :param int length: Cell length in pixels
    :param int width: Cell width in pixels
    :param int separation: An int between (0, `width`) controlling how much pinching is happening.
    :param bool pinching: Controls whether pinching is happening

    Returns
    -------

    cell : np.array
       A numpy array which contains an OPL image of the cell. Can be converted to a mask by just taking ``cell > 0``.

    """

    L = int(np.rint(length))
    W = int(np.rint(width))
    new_cell = np.zeros((L, W))
    R = (W - 1) / 2

    x_cyl = np.arange(0, 2 * R + 1, 1)
    I_cyl = np.sqrt(R ** 2 - (x_cyl - R) ** 2)
    L_cyl = L - W
    new_cell[int(W / 2):-int(W / 2), :] = I_cyl

    x_sphere = np.arange(0, int(W / 2), 1)
    sphere_Rs = np.sqrt((R) ** 2 - (x_sphere - R) ** 2)
    sphere_Rs = np.rint(sphere_Rs).astype(int)

    for c in range(len(sphere_Rs)):
        R_ = sphere_Rs[c]
        x_cyl = np.arange(0, R_, 1)
        I_cyl = np.sqrt(R_ ** 2 - (x_cyl - R_) ** 2)
        new_cell[c, int(W / 2) - sphere_Rs[c]:int(W / 2) + sphere_Rs[c]] = np.concatenate((I_cyl, I_cyl[::-1]))
        new_cell[L - c - 1, int(W / 2) - sphere_Rs[c]:int(W / 2) + sphere_Rs[c]] = np.concatenate((I_cyl, I_cyl[::-1]))

    if separation > 2 and pinching:
        S = int(np.rint(separation))
        new_cell[int((L - S) / 2) + 1:-int((L - S) / 2) - 1, :] = 0
        for c in range(int((S+1) / 2)):
            R__ = sphere_Rs[-c - 1]
            x_cyl_ = np.arange(0, R__, 1)
            I_cyl_ = np.sqrt(R__ ** 2 - (x_cyl_ - R__) ** 2)
            new_cell[int((L-S) / 2) + c + 1, int(W / 2) - R__:int(W / 2) + R__] = np.concatenate((I_cyl_, I_cyl_[::-1]))
            new_cell[-int((L-S) / 2) - c - 1, int(W / 2) - R__:int(W / 2) + R__] = np.concatenate((I_cyl_, I_cyl_[::-1]))
    new_cell = new_cell.astype(int)
    return new_cell


@njit
def OPL_to_FL(cell, density):
    """

    :param np.ndarray cell: A 2D numpy array consisting of a rasterised cell
    :param float density: Number of fluorescent molecules per volume element to sample in the cell
    :return: A cell with fluorescent reporters sampled in it
    :rtypes: np.ndarray
    """

    cell_normalised = (cell/cell.sum())
    i, j = np.arange(cell_normalised.shape[0]), np.arange(cell_normalised.shape[1])
    indices = [] #needed for njit
    for ii in i:
        for jj in j:
            indices.append((ii,jj))
    weights = cell_normalised.flatten()
    n_molecules = int(density * np.sum(cell))
    choices = np.searchsorted(np.cumsum(weights), np.random.rand(n_molecules)) # workaround for np.random.choice from
    FL_cell = np.zeros(cell.shape)
    for c in choices:
        FL_cell[indices[c]] += 1
    return FL_cell


@njit
def generate_deviation_from_CL(centreline, thickness):
    return np.arange(thickness) + centreline - int(np.ceil(thickness ))

@njit
def gen_3D_coords_from_2D(test_cells, centreline, thickness):
    return np.where(test_cells == thickness) + (generate_deviation_from_CL(centreline, thickness),)

@njit
def convert_to_3D_numba(cell):
    expanded_scene = cell
    volume_shape = expanded_scene.shape[0:] + (int(expanded_scene.max()*2),)
    test_cells = rounder(expanded_scene)
    centreline = int(expanded_scene.max() )
    cells_3D = np.zeros(volume_shape,dtype = np.ubyte)
    for t in range(int(expanded_scene.max() *2 )):
        test_coords = gen_3D_coords_from_2D(test_cells, centreline, t)
        for x, y in zip(test_coords[0], (test_coords[1])):
            for z in test_coords[2]:
                cells_3D[x, y, z] = 1
    return cells_3D

@njit
def rounder(x):
    out = np.empty_like(x)
    np.round(x, 0, out)
    return out

def convert_to_3D(cell):
    cells_3D = convert_to_3D_numba(cell)
    cells_3D = np.moveaxis(cells_3D, -1, 0)
    cells_3D[cells_3D.shape[0]//2:,:, :] = cells_3D[:cells_3D.shape[0]//2,:, :][::-1]
    return cells_3D

def draw_scene_from_segments(cells_segment_data, space_size, offset, label_masks, cell_texture=False):
    """
    Draw an OPL scene and mask from segment chain data.

    Each cell is a chain of overlapping spheres. The OPL at each pixel is
    2 * max(sqrt(R_i^2 - d_i^2)) over all segments i of that cell, where
    d_i is the distance from the pixel to segment centre i.

    Parameters
    ----------
    cells_segment_data : list of dict
        Per-cell dicts with keys:
            'positions': np.array (N,2) — segment (x,y) centres
            'radii': np.array (N,) — segment radii
            'mask_label': int
            'cell_id': int
    space_size : tuple
        (height, width) of output arrays.
    offset : int
        Pixel offset added to all positions.
    label_masks : bool
        If True, mask pixels get the cell's mask_label value.
        If False, mask is boolean.
    cell_texture : bool or dict
        If truthy, apply Perlin noise texture. If dict, passed as
        kwargs to apply_cell_texture().

    Returns
    -------
    scene : np.ndarray
        2D OPL image.
    mask : np.ndarray
        2D labelled or boolean mask.
    """
    space_size = np.array(space_size)
    # Overlap ownership buffers:
    # - owner_opl stores the winning per-pixel OPL value
    # - owner_label stores the label of the winning cell
    # This avoids zeroing overlap pixels (which creates artificial cracks).
    owner_opl = np.zeros(space_size, dtype=np.float64)
    owner_label = np.zeros(space_size, dtype=np.int32)

    for cell_data in cells_segment_data:
        positions = cell_data['positions']  # (N, 2)
        radii = cell_data['radii']          # (N,)
        sim_mask_label = cell_data['mask_label']
        cell_id = cell_data['cell_id']

        if len(positions) == 0:
            continue

        # Compute bounding box for this cell's segments
        max_r = np.max(radii)
        min_x = np.min(positions[:, 0]) - max_r + offset
        max_x = np.max(positions[:, 0]) + max_r + offset
        min_y = np.min(positions[:, 1]) - max_r + offset
        max_y = np.max(positions[:, 1]) + max_r + offset

        # Clip to image bounds
        y_start = max(0, int(np.floor(min_y)))
        y_end = min(space_size[0], int(np.ceil(max_y)) + 1)
        x_start = max(0, int(np.floor(min_x)))
        x_end = min(space_size[1], int(np.ceil(max_x)) + 1)

        if y_start >= y_end or x_start >= x_end:
            continue

        # Pixel grid for this cell's bounding box
        ys = np.arange(y_start, y_end, dtype=np.float64)
        xs = np.arange(x_start, x_end, dtype=np.float64)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')

        # For each segment, compute OPL contribution via vectorised operations
        opl_patch = np.zeros_like(grid_y)
        for seg_idx in range(len(positions)):
            cx = positions[seg_idx, 0] + offset
            cy = positions[seg_idx, 1] + offset
            r = radii[seg_idx]
            d_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
            r_sq = r * r
            inside = d_sq < r_sq
            z = np.zeros_like(d_sq)
            z[inside] = np.sqrt(r_sq - d_sq[inside])
            np.maximum(opl_patch, z, out=opl_patch)

        # OPL is 2 * max hemisphere height (full sphere projection)
        cell_opl = 2.0 * opl_patch

        # Apply texture if requested
        if cell_texture and np.any(cell_opl > 0):
            texture_params = cell_texture if isinstance(cell_texture, dict) else {}
            # Create a tight crop for texture, then paste back
            nz_rows, nz_cols = np.nonzero(cell_opl > 0)
            if len(nz_rows) > 0:
                tr0, tr1 = nz_rows.min(), nz_rows.max() + 1
                tc0, tc1 = nz_cols.min(), nz_cols.max() + 1
                crop = cell_opl[tr0:tr1, tc0:tc1]
                crop = apply_cell_texture(crop, cell_id, **texture_params)
                cell_opl[tr0:tr1, tc0:tc1] = crop

        owner_patch = owner_opl[y_start:y_end, x_start:x_end]
        label_patch = owner_label[y_start:y_end, x_start:x_end]

        # Winner-takes-all ownership in contested pixels.
        claim = cell_opl > owner_patch
        owner_patch[claim] = cell_opl[claim]
        label_patch[claim] = sim_mask_label

    scene = owner_opl

    if label_masks:
        mask = owner_label
    else:
        mask_borders = find_boundaries(owner_label, mode="thick", connectivity=2)
        mask = np.where(mask_borders, 0, owner_label)
        mask = opening(mask)
        mask = mask.astype(bool)

    # For binary masks only, keep scene limited to final mask area.
    if not label_masks:
        scene = scene * mask.astype(bool)
    return scene, mask


def scene_plotter(scene_array, output_dir, name, a, matplotlib_draw):
    if matplotlib_draw == True:
        plt.figure(figsize=(3, 10))
        plt.imshow(scene_array)
        plt.tight_layout()
        plt.savefig(output_dir + "/{}_{}.png".format(name, str(a).zfill(3)))
        plt.clf()
        plt.close('all')
    else:
        im = Image.fromarray(scene_array.astype(np.uint8))
        im.save(output_dir + "/{}_{}.tif".format(name, str(a).zfill(3)))


def make_images_same_shape(real_image, synthetic_image, rescale_int=True):
    """ Makes a synthetic image the same shape as the real image """

    assert real_image.shape[0] < synthetic_image.shape[
        0], "Real image has a higher diemsion on axis 0, increase y_border_expansion_coefficient"
    assert real_image.shape[1] < synthetic_image.shape[
        1], "Real image has a higher diemsion on axis 1, increase x_border_expansion_coefficient"

    x_diff = synthetic_image.shape[1] - real_image.shape[1]
    remove_from_left, remove_from_right = div_odd(x_diff)
    y_diff = synthetic_image.shape[0] - real_image.shape[0]
    if real_image.shape[1] % 2 == 0:
        if synthetic_image.shape[1] % 2 == 0:
            if y_diff > 0:
                synthetic_image = synthetic_image[y_diff:, remove_from_left - 1:-remove_from_right]
            else:
                synthetic_image = synthetic_image[:, remove_from_left:-remove_from_right]
                real_image = real_image[abs(y_diff):, :]
        elif synthetic_image.shape[1] % 2 == 1:
            if y_diff > 0:
                synthetic_image = synthetic_image[y_diff:, remove_from_left:-remove_from_right]
            else:
                synthetic_image = synthetic_image[:, remove_from_left:-remove_from_right]
                real_image = real_image[abs(y_diff):, :]
    elif real_image.shape[1] % 2 == 1:
        if synthetic_image.shape[1] % 2 == 0:
            if y_diff > 0:
                synthetic_image = synthetic_image[y_diff:, remove_from_left:-remove_from_right]
            else:
                synthetic_image = synthetic_image[:, remove_from_left:-remove_from_right]
                real_image = real_image[abs(y_diff):, :]
        elif synthetic_image.shape[1] % 2 == 1:
            if y_diff > 0:
                synthetic_image = synthetic_image[y_diff:, remove_from_left - 1:-remove_from_right]
            else:
                synthetic_image = synthetic_image[:, remove_from_left:-remove_from_right]
                real_image = real_image[abs(y_diff):, :]

    if rescale_int:
        real_image = rescale_intensity(real_image.astype(np.float32), out_range=(0, 1))
        synthetic_image = rescale_intensity(synthetic_image.astype(np.float32), out_range=(0, 1))
    return real_image, synthetic_image


def get_space_size_from_segments(cell_timeseries_segments, offset=30):
    """
    Compute canvas size from segment-chain timeseries data.

    Parameters
    ----------
    cell_timeseries_segments : list of list of dict
        Per-frame, per-cell segment data dicts with 'positions' and 'radii'.
    offset : int
        Pixel offset applied to positions.

    Returns
    -------
    tuple
        (height, width) for the canvas.
    """
    max_y, max_x = 0, 0
    for frame_data in cell_timeseries_segments:
        for cell_data in frame_data:
            positions = cell_data['positions']
            radii = cell_data['radii']
            if len(positions) == 0:
                continue
            max_r = np.max(radii)
            frame_max_x = np.max(positions[:, 0]) + max_r + offset
            frame_max_y = np.max(positions[:, 1]) + max_r + offset
            if frame_max_x > max_x:
                max_x = frame_max_x
            if frame_max_y > max_y:
                max_y = frame_max_y
    return (int(1.2 * max_y), int(1.5 * max_x))


def clean_up_mask(mask):
    return remove_small_objects(label(mask))
