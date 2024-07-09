import unittest
from SyMBac.drawing import generate_curve_props, gen_cell_props_for_draw, get_space_size
from SyMBac.phase_contrast_drawing import *
from SyMBac.PSF import get_phase_contrast_kernel, get_condensers
from SyMBac.misc import get_sample_images
import pymunk
from tqdm import tqdm

import importlib

if importlib.util.find_spec("cupy") is None:
    manual_update = True
else:
    manual_update = False


class TestExample(unittest.TestCase):

    def test_partial_example(self):
        real_image = get_sample_images()["E. coli 100x"]

        condensers = get_condensers()
        W, R, diameter = condensers["Ph3"]
        radius = 50
        # F = 5
        λ = 0.75
        resize_amount = 3
        pix_mic_conv = 0.0655  ##0.108379937 micron/pix for 60x, 0.0655 for 100x
        scale = pix_mic_conv / resize_amount
        min_sigma = 0.42 * 0.6 / 6 / scale  # micron#
        sigma = min_sigma
        NA = 1.45
        n = 1.4
        kernel_params = (R, W, radius, scale, NA, n, sigma, λ)

        sim_length = 5
        cell_timeseries, space = run_simulation(
            trench_length=15,
            trench_width=1.5,
            cell_max_length=6,  # 6, long cells # 1.65 short cells
            cell_width=1,  # 1 long cells # 0.95 short cells
            sim_length=sim_length,
            pix_mic_conv=pix_mic_conv,
            gravity=0,
            phys_iters=20,
            max_length_var=3,
            width_var=0.3,
            save_dir="/tmp/"
        )  # growth phase
        main_segments = get_trench_segments(space)
        ID_props = generate_curve_props(cell_timeseries)
        cell_timeseries_properties = Parallel(n_jobs=-1)(
            delayed(gen_cell_props_for_draw)(a, ID_props) for a in tqdm(cell_timeseries, desc='Timeseries Properties'))
        do_transformation = True
        offset = 30
        mask_threshold = 12
        label_masks = True
        space_size = get_space_size(cell_timeseries_properties)
        scenes = Parallel(n_jobs=13)(delayed(draw_scene)(
            cell_properties, do_transformation, space_size, offset, label_masks) for cell_properties in
                                     tqdm(cell_timeseries_properties, desc='Scene Draw:'))

        media_multiplier = 30
        cell_multiplier = 1
        device_multiplier = -50
        y_border_expansion_coefficient = 2
        x_border_expansion_coefficient = 2

        temp_expanded_scene, temp_expanded_scene_no_cells, temp_expanded_mask = generate_PC_OPL(
            main_segments=main_segments,
            offset=offset,
            scene=scenes[0][0],
            mask=scenes[0][1],
            media_multiplier=media_multiplier,
            cell_multiplier=cell_multiplier,
            device_multiplier=cell_multiplier,
            y_border_expansion_coefficient=y_border_expansion_coefficient,
            x_border_expansion_coefficient=x_border_expansion_coefficient,
            fluorescence=False,
            defocus=30
        )

        temp_kernel = get_phase_contrast_kernel(*kernel_params);
        convolved = convolve_rescale(temp_expanded_scene, temp_kernel, 1 / resize_amount, rescale_int=True);
        real_resize, expanded_resized = make_images_same_shape(real_image, convolved, rescale_int=True);

        self.assertIsInstance(space, pymunk.space.Space)
        self.assertEqual(real_resize.shape[0], expanded_resized.shape[0])
        self.assertEqual(real_resize.shape[1], expanded_resized.shape[1])


if __name__ == "__main__":
    unittest.main()
