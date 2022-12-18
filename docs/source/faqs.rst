FAQs
====

- **Do I need to have a GPU?**

  - No, although image synthesis will be around 40x slower on the CPU. SyMBac will detect that you do not have CuPy installed and default to using CPU convolution.
  - Interactive image optimisation will be very painful on the CPU. By default I turn off slider interactivity if you are using the CPU, so that you can move a slider without the CPU being maxed out. This means that every time you move a slider you must click the button to update the image (do a convolution).
- **Can I generate fluorescence images as well?**
  
  - Yes, you can do fluorescence image generation, just make sure that in the interactive image generation part of the code, you select fluorescence.
  - Since our fluorescence kernel is defined to be a subset of the phase contrast kernel, you can choose **any** condenser, and your fluorescence kernel should be correct. Just ensure that the imaging wavelength, numerical aperture, refractive index, and pixel size are set correctly.
- **What format do my images need to be in?**
  
  - The real images you are trying to replicate should be in the format of single-trench timeseries images. If you are unsure what this is, you can call ``get_sample_images()["E. coli 100x"]`` from ``SyMBac.misc`` for an example image.
- **I'm getting libGL MESA-LOADER errors**

  - See this `StackExchange link <https://unix.stackexchange.com/questions/655495/trying-to-run-pygame-on-my-conda-environment-on-my-fresh-manjaro-install-and-ge>`_.

- **I'm getting a libGL error**

  - E.g:
  
  .. code-block:: bash
    
      libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri) libGL error: failed to load driver: swrast

  
  - Try running:

  .. code-block:: bash

    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
