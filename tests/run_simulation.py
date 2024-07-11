import os
os.environ['DISPLAY'] = ':1'

from symbac.simulation import Simulation


my_simulation = Simulation(
    trench_length=15,
    trench_width=2.15,
    cell_max_length=6.65, #6, long cells # 1.65 short cells
    cell_width= 1, #1 long cells # 0.95 short cells
    sim_length = 100,
    pix_mic_conv = 0.065,
    gravity=0,
    phys_iters=15,
    max_length_var = 0.,
    width_var = 0.,
    lysis_p = 0.00,
    save_dir=".",
    resize_amount = 3,
)

my_simulation.run_simulation(show_window = True)
