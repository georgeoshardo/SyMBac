import streamlit as st
import pandas as pd
import numpy as np
from SyMBac.simulation import Simulation
from SyMBac.PSF import PSF_generator
from SyMBac.renderer import Renderer
from SyMBac.PSF import Camera
from SyMBac.misc import get_sample_images
real_image = get_sample_images()["E. coli 100x"]


st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('SyMBac Web Server')

#global_settings_tab, simulation_tab, PSF_tab = st.tabs(["Global Settings", "Simulation", "PSF"])

#with global_settings_tab:
#
#    st.header("Global Settings", anchor=None)#
#
#    pix_mic_conv = float(st.text_input('Pixel size (micron/pixel)',  value=0.065))
#    resize_amount = float(st.text_input('Simulation scale factor (resize amount, suggested = 3)',  value=3))
#with simulation_tab:
#    st.header("Simulation Settings", anchor=None)
#    trench_length = float(st.text_input('Trench Length (micron)',  value=15))
#    trench_width = float(st.text_input('Trench Width (micron)',  value=1.3))
#    cell_max_length = float(st.text_input('Maximum cell length (micron)',  value=6.65))
#    cell_width = float(st.text_input('Cell Width (micron)',  value=1))
#    sim_length = int(st.text_input("Simulation length (timesteps)", value = 100))
    


#    my_simulation = Simulation(
#        trench_length=trench_length,
#        trench_width=trench_width,
#        cell_max_length=cell_max_length, #6, long cells # 1.65 short cells
 #       cell_width= cell_width, #1 long cells # 0.95 short cells
 #       sim_length = sim_length,
 #       pix_mic_conv = pix_mic_conv,
  #      gravity=0,
   #     phys_iters=15,
    #    max_length_var = 0.,
     #   width_var = 0.,
      #  lysis_p = 0.,
       # save_dir="/tmp/",
       # resize_amount = resize_amount
    #)

    #st.button("Run simulation", on_click = my_simulation.run_simulation, args=(False, True))

with PSF_tab:
    

    st.header("PSF Settings", anchor=None)

    PSF_type = st.selectbox("PSF type", options=["Simple Fluorescence", "Phase Contrast"])

    radius = int(st.text_input("Radius", value=100))
    wavelength = float(st.text_input('Wavelength (micron)',  value=0.4))
    NA = float(st.text_input("Numerical Aperture", value=1.49))
    n = float(st.text_input("Refractive Index", value=1.51))


    if PSF_type == "Simple Fluorescence":
        my_kernel = PSF_generator(
            radius = radius,
            wavelength = wavelength,
            NA = NA,
            n = n,
            resize_amount = 3,
            pix_mic_conv = 0.065,
            apo_sigma = None,
            mode="simple fluo")
        my_kernel.calculate_PSF()
        #st.pyplot(my_kernel.plot_PSF(), clear_figure=True, figsize=10)
        st.pyplot(my_kernel.plot_PSF())
    elif PSF_type == "Phase Contrast":
        apo_sigma = float(st.slider("Apodisation Sigma", value=10.0, min_value=0.01, max_value=30.0))
        condenser = st.selectbox("Condenser type", options=["Ph1", "Ph2", "Ph3", "Ph4", "PhF"])
        my_kernel = PSF_generator(
            radius = radius,
            wavelength = wavelength,
            NA = NA,
            n = n,
            resize_amount = 3,
            pix_mic_conv = 0.065,
            apo_sigma = apo_sigma,
            mode="phase contrast",
            condenser = condenser
        )
        my_kernel.calculate_PSF()

        st.pyplot(my_kernel.plot_PSF())
        #st.image(my_kernel.kernel, caption="PSF", clamp=True)
