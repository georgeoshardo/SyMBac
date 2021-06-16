def f():
    import napari
    import sys
    sys.path.insert(0,'/home/georgeos/Documents/GitHub/SYMPTOMM2')
    from SYMPTOMM.cell import Cell
    from SYMPTOMM.scene_functions import create_space, step_and_update
    from SYMPTOMM.trench_geometry import trench_creator
    from SYMPTOMM.phase_contrast_drawing import draw_scene, run_simulation, get_trench_segments
    from SYMPTOMM.general_drawing import generate_curve_props
    from SYMPTOMM.OPL_generation import gen_cell_props_for_draw
    from joblib import Parallel, delayed
    import tifffile
    import numpy as np
    from skimage.exposure import rescale_intensity
    from skimage.transform import rescale, resize, downscale_local_mean
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets
    from skimage.exposure import rescale_intensity
    from ipywidgets import interactive
    import os
    from PIL import Image
    import skimage
    from tqdm.notebook import tqdm
    import gc
    from SYMPTOMM.PSF import get_phase_contrast_kernel, get_condensers
    
    condensers = get_condensers()
    W, R, diameter = condensers["Ph2"]
    radius=50
    #F = 5
    λ = 0.75
    resize_amount = 3
    pix_mic_conv = 0.0655 ##0.108379937 micron/pix for 60x, 0.0655 for 100x
    scale = pix_mic_conv / resize_amount 
    min_sigma = 0.42*0.6/6 / scale # micron#
    sigma=min_sigma
    NA=1.54
    n = 1.4
    kernel_params = (R,W,radius,scale,NA,n,sigma,λ)

#kernel_params = (R,W,radius,scale,F,sigma,λ)

    
    sim_length = 3
    cell_timeseries, space = run_simulation(
        trench_length=17, 
        trench_width=1.5, 
        cell_max_length=4.7, 
        cell_width=1.15, 
        sim_length = sim_length, 
        pix_mic_conv = pix_mic_conv,
        gravity=-10,
        phys_iters=25
            ) # growth phase
    main_segments = get_trench_segments(space)
    ID_props = generate_curve_props(cell_timeseries)
    cell_timeseries_properties = Parallel(n_jobs=14)(
        delayed(gen_cell_props_for_draw)(a, ID_props) for a in tqdm(cell_timeseries, desc='Timeseries Properties'))
    do_transformation = True
    offset = 30
    mask_threshold = 18
    label_masks = True
    space_size = get_space_size(cell_timeseries_properties)
    
    real_image = tifffile.imread("/home/georgeos/Storage/Dropbox (Cambridge University)/PhD_Georgeos_Hardo/ML_based_segmentation_results/40x_Ph2_test_1.5/top_trenches_PC/trench_{}/T_{}.tif".format(
        str(np.random.randint(1,56)).zfill(2),
        str(np.random.randint(10,20)).zfill(3)))
    import numpy as np
    Parallel(n_jobs=-1)(delayed(draw_scene)(
        cell_properties, do_transformation, mask_threshold, space_size, offset, label_masks) for cell_properties in tqdm(cell_timeseries_properties, desc='Scene Draw:'))
f()
