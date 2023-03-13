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
