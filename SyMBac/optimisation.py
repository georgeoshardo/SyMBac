from ipywidgets import interactive, fixed
import importlib
from SyMBac.phase_contrast_drawing import generate_test_comparison

if importlib.util.find_spec("cupy") is None:
    manual_update = True
    print("No CuPy installation detected. Will default to manual updating of manual optimiser")
else:
    manual_update = False
    

def manual_optimise(scenes, scale, offset, main_segments, kernel_params, min_sigma, resize_amount, real_image, image_params, x_border_expansion_coefficient, y_border_expansion_coefficient):
    """
    Generate an interactive iPython cell to manually optimise the image similarity between the simulation and a real image.

    Parameters
    ----------
    scenes: list(tuple(numpy.ndarray, numpy.ndarray))
        A list of tuples, with each tuple containing an OPL image of cells, and the corresponding masks.
    scale: 
        The micron/pixel value of the image at the scale of the simulation. Typically scale = pix_mic_conv / resize_amount.
    offset: 
        This is a parameter which ensures that cells never touch the edge of the image after being transformed. In general this can be left as is (30), but you will recieve an error if it needs increasing.
    main_segments: pandas.core.frame.DataFrame
        A pandas dataframe contianing information about the trench position and geometry for drawing. Generated using :func:`~SyMBac.phase_contrast_drawing.get_trench_segments`
    kernel_params: tuple
        ``(R,W,radius,scale,NA,n,sigma,λ)``, arguments for :func:`~SyMBac.PSF.get_phase_contrast_kernel`
    min_sigma: float
        Theoretical minimum sigma for gaussian apodisation of the PSF, if unsure, leave at 2.
    resize_amount: int
        The scaling factor which the simulation was run at. Typically chosen to be 3.
    real_image: numpy.ndarray
        A sample of a real image to optimise against.
    image_params: tuple
        ``(real_media_mean, real_cell_mean, real_device_mean, real_means, real_media_var, real_cell_var, real_device_var, real_vars)`` Generated from the labeling of the device, media, and cells of the real image using Napari. 
    y_border_expansion_coefficient: float
        A multiplier to fractionally scale the y dimension of the image by. Generally good to make this bigger than 1.5.
    x_border_expansion_coefficient: float
        A multiplier to fractionally scale the x dimension of the image by.

    Returns
    -------
    params: ipywidgets.widgets.interaction.interactive
        An interactive object (to be displayed within an iPython notebook such as Jupyter Lab). 
        The object can be modified using the sliders, which will automatically update the attributes of the object, without needing to save it. The return object is then passed to :func:`~SyMBac.phase_contrast_drawing.generate_training_data`.
    """
    
    if manual_update:
        print("No CuPy installation detected. Will default to manual updating of manual optimiser")

    
    mean_error = []
    media_error = []
    cell_error = []
    device_error = []

    mean_var_error = []
    media_var_error = []
    cell_var_error = []
    device_var_error = []

    error_params = (mean_error,media_error,cell_error,device_error,mean_var_error,media_var_error,cell_var_error,device_var_error)

    params = interactive(
        generate_test_comparison,
        {'manual': manual_update},
        media_multiplier=(-300,300,1),
        cell_multiplier=(-30,30,0.01),
        device_multiplier=(-300,300,1),
        sigma=(min_sigma,min_sigma*20, min_sigma/20),
        scene_no = (0,len(scenes)-1,1),
        noise_var=(0,0.001, 0.0001),
        scale=fixed(scale),
        match_fourier = [True, False],
        match_histogram = [True, False],
        match_noise = [True, False],
        offset=fixed(offset),
        main_segments = fixed(main_segments),
        debug_plot=fixed(True),
        scenes = fixed(scenes),
        kernel_params = fixed(kernel_params),
        resize_amount = fixed(resize_amount), 
        real_image = fixed(real_image),
        image_params = fixed(image_params),
        error_params = fixed(error_params),
        x_border_expansion_coefficient = fixed(x_border_expansion_coefficient),
        y_border_expansion_coefficient = fixed(y_border_expansion_coefficient),
        fluorescence=[False, True],
        defocus=(0,20,0.1)
    );
    return params

