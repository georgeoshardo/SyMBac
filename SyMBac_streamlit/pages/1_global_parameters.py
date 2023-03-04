import streamlit as st
import pickle

def number_input(label, key, min_value=None, max_value=None, step=0.01, format_=None):
    value = st.number_input(label = label, min_value = min_value, max_value = max_value, step = step, format = format_)
    if key not in st.session_state:
        st.session_state[key] = value

st.set_page_config(page_title="Global paramters", page_icon="📈")

uploaded_simulation = st.file_uploader("Upload Previous Simulation (no need to define previous values)")
if uploaded_simulation:
    st.session_state["my_simulation"] = pickle.load(uploaded_simulation)
    st.session_state['pix_mic_conv'] = st.session_state["my_simulation"].pix_mic_conv
    st.session_state["resize_amount"] = st.session_state["my_simulation"].resize_amount


number_input(label = "Pixel size ($\mu m/pix$)", key = "pix_mic_conv", step=0.001, format_="%.3f")
number_input("Simulation scale factor", "resize_amount")



#pix_mic_conv = st.number_input('Pixel size (micron/pixel)')
#st.session_state["pix_mic_conv"] = pix_mic_conv
#resize_amount = st.number_input('Simulation scale factor (resize amount, suggested = 3)')
#st.session_state["resize_amount"] = resize_amount
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 