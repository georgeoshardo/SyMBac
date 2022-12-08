import CellModeller
import noise
from CellModeller.Simulator import Simulator
import os
import numpy as np
from glob import glob
import pickle
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed
from PIL import Image
from skimage.transform import rotate
from natsort import natsorted
from SyMBac.drawing import raster_cell, OPL_to_FL, clean_up_mask


class ColonySimulation:
    def __init__(self, cellmodeller_model, max_cells, pix_mic_conv, resize_amount, save_dir):
        self.cellmodeller_model = cellmodeller_model
        self.max_cells = max_cells
        self.pix_mic_conv = pix_mic_conv
        self.resize_amount = resize_amount
        self.save_dir = save_dir
        self.n_simulations = 1
        self.masks_dir = f"data/masks/"
        self.OPL_dir = f"data/scenes/"
        try:
            os.mkdir("data/")
            print(f"Creating directory data/{self.save_dir}")
        except:
            pass
        try:
            os.mkdir(self.masks_dir)
        except:
            pass
        try:
            os.mkdir(self.OPL_dir)
        except:
            pass


    def run_cellmodeller_sim(self, num_sim):
        for n in range(num_sim):
            try:
                os.mkdir(f"data/{self.save_dir}")
            except:
                pass
            try:
                os.mkdir(self.save_dir+str(self.n_simulations))
            except:
                pass
            self.simulation = Simulator(self.cellmodeller_model, dt = 0.025, clPlatformNum=0, clDeviceNum=0, saveOutput=True, pickleSteps=1,
                            outputDirName=f"{self.save_dir}/{str(self.n_simulations)}", is_gui=True)

            # Run the simulation to ~n cells
            while len(self.simulation.cellStates) < self.max_cells:
                self.simulation.step()
            self.n_simulations += 1

    def get_cellmodeller_properties(self, cellmodeller_pickle):
        properties = []
        for cell in cellmodeller_pickle["cellStates"].values():
            angle = np.rad2deg(
                np.arctan2(cell.ends[1][1] - cell.ends[0][1], cell.ends[1][0] - cell.ends[0][0])) + 180 + 90
            properties.append(
                [
                    (cell.length + 2 * cell.radius) / self.pix_mic_conv * self.resize_amount,
                    cell.radius * 2 / self.pix_mic_conv * self.resize_amount,
                    angle,
                    [x / self.pix_mic_conv * self.resize_amount for x in cell.pos[:-1]]
                ]
            )
        return properties

    def get_simulation_dirs(self):
        self.simulation_dirs = natsorted(glob(f"data/{self.save_dir}/*"))
        return self.simulation_dirs

    def get_simulation_pickles(self):
        pickles = []
        for dir in self.get_simulation_dirs():
            pickles.append(
                natsorted(glob(f"{dir}/*.pickle"))
            )
        self.pickles = natsorted(pickles)
        self.pickles_flat = natsorted([item for sublist in self.pickles for item in sublist])
        return self.pickles

    def pickle_opener(self, dir):
        return pickle.load(open(dir, "rb"))

    def get_scene_size(self, cellmodeller_properties):
        cell_positions = np.array([x[3] for x in (cellmodeller_properties)])
        max_cell_length = np.array([x[0] for x in cellmodeller_properties]).max().astype(int)
        max_x, min_x = cell_positions[:, 0].max(), cell_positions[:, 0].min()
        max_y, min_y = cell_positions[:, 1].max(), cell_positions[:, 1].min()
        scene_shape = (np.ceil(max_y + abs(min_y) + max_cell_length * 2).astype(int),
                       np.ceil(max_x + abs(min_x) + max_cell_length * 2).astype(int))

        return scene_shape

    def get_max_scene_size(self):
        pickles_flat = natsorted([item for sublist in self.pickles for item in sublist])
        scene_shapes = []

        for _ in pickles_flat:
            cellmodeller_properties = self.get_cellmodeller_properties(self.pickle_opener(_))

            scene_shapes.append(self.get_scene_size(cellmodeller_properties))
        self.scene_shape = tuple(np.array(scene_shapes).max(axis=0))
        return self.scene_shape

    def draw_scene(self, cellmodeller_properties, save = False, savename = False, return_save = False, FL = False, density = 1, random_distribution = "uniform", distribution_args = (1,1)):
        space = np.zeros(self.scene_shape)
        mask = np.zeros(self.scene_shape)
        mask_counter = 1
        if random_distribution == "uniform":
            density_modifiers = [np.random.uniform(*distribution_args) for _ in range(len(cellmodeller_properties))]
        for c in range(len(cellmodeller_properties)):
            position = cellmodeller_properties[c][3]
            offset = np.ceil(np.mean(self.scene_shape) / 2).astype(int)
            x = np.array(position).astype(int)[0] + offset
            y = np.array(position).astype(int)[1] + offset
            OPL_cell = raster_cell(cellmodeller_properties[c][0], cellmodeller_properties[c][1], separation=0)
            rotated_OPL_cell = rotate(OPL_cell, - (cellmodeller_properties[c][2]), resize=True, clip=False,
                                      preserve_range=True, center=(x, y))
            cell_y, cell_x = (np.array(rotated_OPL_cell.shape) / 2).astype(int)
            offset_y = rotated_OPL_cell.shape[0] - space[y - cell_y:y + cell_y, x - cell_x:x + cell_x].shape[0]
            offset_x = rotated_OPL_cell.shape[1] - space[y - cell_y:y + cell_y, x - cell_x:x + cell_x].shape[1]
            if FL:
                FL_cell = OPL_to_FL(rotated_OPL_cell, density*density_modifiers[c])
                space[
                    y - cell_y:y + cell_y + offset_y,
                    x - cell_x:x + cell_x + offset_x
                ] += FL_cell
            else:
                space[
                    y - cell_y:y + cell_y + offset_y,
                    x - cell_x:x + cell_x + offset_x
                ] += (rotated_OPL_cell)
            mask[
                y - cell_y:y + cell_y + offset_y,
                x - cell_x:x + cell_x + offset_x
            ] += ((rotated_OPL_cell) > 1) * mask_counter
            mask_counter += 1
        mask =  clean_up_mask(mask)
        if save and not savename:
            raise Exception("Add a savename if you wish to save")
        if save and savename:
            im = Image.fromarray(space.astype(np.uint16))
            im.save(f"data/scenes/{savename}.png")
            im = Image.fromarray(mask.astype(np.uint16))
            im.save(f"data/masks/{savename}.png")
            if return_save:
                return space, mask
            else:
                return None
        else:
            return space, mask

    def draw_OPL_from_idx(self,idx, FL = False, density = 1, random_distribution = "uniform", distribution_args = (0.99, 1.01)):
        cellmodeller_properties = self.get_cellmodeller_properties(
            self.pickle_opener(
                self.pickles_flat[idx]
            )
        )
        return self.draw_scene(cellmodeller_properties, False, False, False, FL, density, random_distribution, distribution_args)

    def draw_simulation_OPL(self, n_jobs, FL = False, density = 1, random_distribution = "uniform", distribution_args = (0.99,1.01)):

        self.get_simulation_dirs()
        self.get_max_scene_size()

        all_cellmodeller_properties = [self.get_cellmodeller_properties(self.pickle_opener(_)) for _ in self.pickles_flat]

        n_files = len(glob("data/scenes/*.png"))
        zero_pads = np.ceil(np.log10(len(self.pickles_flat))).astype(int) + 2
        Parallel(n_jobs=n_jobs)(delayed(self.draw_scene)(_, True, str(i+1+n_files).zfill(zero_pads), False, FL, density, random_distribution, distribution_args) for i, _ in tqdm(enumerate(all_cellmodeller_properties), desc='Scene Draw:'))

