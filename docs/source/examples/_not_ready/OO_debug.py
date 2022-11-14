from SyMBac.cell_simulation import Simulation

my_simulation = Simulation(
    trench_length=15,
    trench_width=1.3,
    cell_max_length=6.65, #6, long cells # 1.65 short cells
    cell_width= 1, #1 long cells # 0.95 short cells
    sim_length = 50,
    pix_mic_conv = 0.065,
    gravity=0,
    phys_iters=5,
    max_length_var = 0.,
    width_var = 0.,
    lysis_p = 0.,
    save_dir="/tmp/"
)

my_simulation.run_simulation(show_window=False)

my_simulation.run_simulation(show_window=False)