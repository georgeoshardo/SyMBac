import streamlit as st
from SyMBac.simulation import Simulation
import matplotlib.pyplot as plt
import pickle
from SyMBac.PSF import PSF_generator
from SyMBac.PSF import Camera


def number_input(label, key, value = None, min_value=None, max_value=None, step=0.01, format_=None):
    value = st.number_input(label = label, value = value, min_value = min_value, max_value = max_value, step = step, format = format_)
    st.session_state[key] = value

st.set_page_config(page_title="PSF")
#st.write(st.session_state["pix_mic_conv"])

PSF_tab, camera_tab = st.tabs(["PSF", "Camera"])

with PSF_tab:
    PSF_type = st.selectbox("PSF type", options=["Simple Fluorescence", "Phase Contrast"])

    number_input("PSF radius", "radius", value = 100, step=1)
    number_input("PSF wavelength ($\mu m$)", "wavelength", value = 0.5, step = 0.01)
    number_input("Numerical Aperture", "NA", value = 1.49, step = 0.01)
    number_input("Refractive Index", "n", value = 1.51, step = 0.01)


    if PSF_type == "Simple Fluorescence":
        my_kernel = PSF_generator(
            radius = st.session_state["radius"],
            wavelength = st.session_state["wavelength"],
            NA = st.session_state["NA"],
            n = st.session_state["n"],
            resize_amount = st.session_state["resize_amount"],
            pix_mic_conv = st.session_state["pix_mic_conv"],
            apo_sigma = None,
            mode="simple fluo")
        my_kernel.calculate_PSF()
        st.session_state["kernel"] = my_kernel
        #st.pyplot(my_kernel.plot_PSF(), clear_figure=True, figsize=10)
        st.pyplot(my_kernel.plot_PSF())
        
    elif PSF_type == "Phase Contrast":
        apo_sigma = float(st.slider("Apodisation Sigma", value=10.0, min_value=0.01, max_value=30.0))
        condenser = st.selectbox("Condenser type", options=["Ph1", "Ph2", "Ph3", "Ph4", "PhF"])
        my_kernel = PSF_generator(
            radius = st.session_state["radius"],
            wavelength = st.session_state["wavelength"],
            NA = st.session_state["NA"],
            n = st.session_state["n"],
            resize_amount = st.session_state["resize_amount"],
            pix_mic_conv = st.session_state["pix_mic_conv"],
            apo_sigma = apo_sigma,
            mode="phase contrast",
            condenser = condenser
        )
        my_kernel.calculate_PSF()
        st.session_state["kernel"] = my_kernel
        st.pyplot(my_kernel.plot_PSF())

with camera_tab:

    number_input("Camera baseline intensity (16 bit)", "baseline", value = 100, step=1)
    number_input("Camera sensitivity ($e^{-}/ADU$)", "sensitivity", value=2.9)
    number_input("Dark noise (dark image pixel variance)", "dark_noise", value = 8.)
    number_input("Sample size", "camera_img_size", value = 150, step = 1)
    st.session_state["my_camera"] = Camera(baseline=st.session_state["baseline"], sensitivity=st.session_state["sensitivity"], dark_noise=st.session_state["dark_noise"])

    st.image(st.session_state["my_camera"].render_dark_image(size=(st.session_state["camera_img_size"],st.session_state["camera_img_size"])))