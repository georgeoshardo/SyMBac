import streamlit as st
from SyMBac.simulation import Simulation
import matplotlib.pyplot as plt
import pickle

def number_input(label, key, value = None, min_value=None, max_value=None, step=0.01, format_=None):
    value = st.number_input(label = label, value = value, min_value = min_value, max_value = max_value, step = step, format = format_)
    st.session_state[key] = value

st.set_page_config(page_title="Simulation")

def customDownloadButton():
    if st.button('Prepare simulation download'):
        data = pickle.dumps(st.session_state["my_simulation"])
        st.download_button('Download simulation', data, file_name='simulation.pickle')

if all(k in st.session_state for k in ("pix_mic_conv","resize_amount")):
    st.header("Run the rigid body simulation")


    number_input("Trench length ($\mu m$)", "trench_length", 15.0)
    number_input("Trench width ($\mu m$)", "trench_width", 1.3)
    number_input("Maximum cell length ($\mu m$)", "cell_max_length", 6.65)
    number_input("Cell width ($\mu m$)", "cell_width", 1.0)
    number_input("Simulation length (timesteps)", "sim_length", value = 100, step = 1)
    pix_mic_conv  = float(st.session_state["pix_mic_conv"])
    resize_amount = float(st.session_state['resize_amount'])
    number_input("Gravity", "gravity", value = 0.0)
    number_input("Physics iterations per timestep", "phys_iters", value = 15, step = 1)
    number_input("Maximum length variance", "max_length_var", value = 0.)
    number_input("Width variance", "width_var", value = 0.)
    number_input("Lysis probability", "lysis_p", value = 0., step = 0.00001, format_="%.5f")


    def sim_runner():
        st.session_state["my_simulation"] = Simulation(
            trench_length=st.session_state["trench_length"],
            trench_width=st.session_state["trench_width"],
            cell_max_length=st.session_state["cell_max_length"], 
            cell_width= st.session_state["cell_width"], 
            sim_length = st.session_state["sim_length"],
            pix_mic_conv = pix_mic_conv,
            gravity=st.session_state["gravity"],
            phys_iters=st.session_state["phys_iters"],
            max_length_var = st.session_state["max_length_var"],
            width_var = st.session_state["width_var"],
            lysis_p = st.session_state["lysis_p"],
            save_dir="/tmp/",
            resize_amount = resize_amount
        )
        st.session_state["my_simulation"].run_simulation(False, True)

    st.button("Run simulation", on_click = sim_runner)
    

    st.header("Draw the simulation OPL images")

    def sim_drawer():
        st.session_state["my_simulation"].draw_simulation_OPL(st.session_state["do_transformation"], st.session_state["label_masks"], True)


    st.selectbox("Transform (bend) cells", [True, False], key="do_transformation")
    st.selectbox("Draw labelled masks", [True, False], key="label_masks")

    st.button("Draw simulation OPL", on_click = sim_drawer)
    
    if "my_simulation" in st.session_state:

        if hasattr(st.session_state["my_simulation"], "masks"):
            
            #@st.cache_resource
            def pickle_dumper():
                return pickle.dumps(st.session_state["my_simulation"])

            def draw_preview(x):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(1,3))
                ax1.imshow(st.session_state["my_simulation"].OPL_scenes[x], cmap="Greys_r")
                ax1.axis("off")
                ax2.imshow(st.session_state["my_simulation"].masks[x])
                ax2.axis("off")
                plt.tight_layout()
                st.pyplot(fig)
            
            #st.download_button("Pickle and download simulation", data=pickle_dumper(), file_name="simulation.pickle")
            customDownloadButton()
            x = st.slider("Preview simulation", min_value=0, max_value=len(st.session_state["my_simulation"].masks)-1, value=0, step=1)
            draw_preview(x)
        

        
else:
    st.write("Please fill in all values in the global parameters page")