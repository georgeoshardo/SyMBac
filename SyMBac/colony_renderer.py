import random
from itertools import cycle
import os

import numpy as np
import noise
from PIL import Image
from glob import glob

from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.transform import rescale
from skimage.util import random_noise
from tqdm.auto import tqdm

from SyMBac.drawing import clean_up_mask
from SyMBac.renderer import convolve_rescale


class ColonyRenderer:
    def __init__(self, simulation, PSF, camera = None):
        self.simulation = simulation
        self.PSF = PSF
        self.camera = camera

        self.scene_shape = simulation.scene_shape
        self.resize_amount = simulation.resize_amount

        self.masks_dir = simulation.masks_dir
        self.OPL_dir = simulation.OPL_dir

        self.mask_dirs = sorted(glob(f"{self.masks_dir}/*.png"))
        self.OPL_dirs = sorted(glob(f"{self.OPL_dir}/*.png"))

    def perlin_generator(self, scale = 5, octaves = 10, persistence = 1.9, lacunarity = 1.8):

        shape = self.scene_shape

        y, x = np.round(shape[0] / self.resize_amount).astype(int), np.round(shape[1] / self.resize_amount).astype(int)

        world = np.zeros((x, y))

        # make coordinate grid on [0,1]^2
        x_idx = np.linspace(0, 1, y)
        y_idx = np.linspace(0, 1, x)
        world_x, world_y = np.meshgrid(x_idx, y_idx)

        # apply perlin noise, instead of np.vectorize, consider using itertools.starmap()
        world = np.vectorize(noise.pnoise2)(world_x / scale,
                                            world_y / scale,
                                            octaves=octaves,
                                            persistence=persistence,
                                            lacunarity=lacunarity)

        # here was the error: one needs to normalize the image first. Could be done without copying the array, though
        img = np.floor((world + .5) * 255).astype(np.uint8)  # <- Normalize world first
        return img

    def random_perlin_generator(self):
        return self.perlin_generator(np.random.uniform(1, 7), np.random.choice([10, 11, 12, 13]), np.random.uniform(1, 1.9),
                                np.random.uniform(1.55, 1.9))


    def OPL_loader(self, idx):
        return np.array(Image.open(self.OPL_dirs[idx]))

    def mask_loader(self, idx):
        return np.array(Image.open(self.mask_dirs[idx]))

    def render_scene(self, idx):
        scene = self.OPL_loader(idx)
        scene = rescale_intensity(scene, out_range=(0, 1))

        temp_kernel = self.PSF.kernel
        temp_kernel = gaussian_filter(temp_kernel, 8.7, mode="reflect")

        convolved = convolve_rescale(scene, temp_kernel, 1/self.resize_amount, rescale_int=True)

        if "phase" in self.PSF.mode.lower():
            bg = self.random_perlin_generator()
            convolved += gaussian_filter(np.rot90(bg)/np.random.uniform(1000,3000), np.random.uniform(1,3), mode="reflect")

        convolved = random_noise((convolved), mode="poisson")
        convolved = random_noise((convolved), mode="gaussian", mean=1, var=0.0002, clip=False)

        convolved = rescale_intensity(convolved.astype(np.float32), out_range=(0, np.iinfo(np.uint16).max)).astype(np.uint16)

        return convolved

    def generate_random_samples(self, n, roll_prob, savedir):
        try:
            os.mkdir(f"{savedir}")
        except:
            pass
        try:
            os.mkdir(f"{savedir}/masks/")
        except:
            pass
        try:
            os.mkdir(f"{savedir}/synth_imgs")
        except:
            pass
        zero_pads = np.ceil(np.log10(n)).astype(int)
        for j, i in tqdm(enumerate(cycle(range(len(self.OPL_dirs)))), total = n):
            sample = self.render_scene(i)
            mask = self.mask_loader(i)
            rescaled_mask =  rescale(mask, 1 / self.resize_amount, anti_aliasing=False, order=0, preserve_range=True).astype(np.uint16)

            if np.random.rand() < roll_prob:
                n_axis_to_roll, amount = random.choice([(0, int(sample.shape[0]/2)), (1, int(sample.shape[1]/2)), ([0,1], (int(sample.shape[0]/2), int(sample.shape[1]/2)))])
                sample = np.roll(sample, amount, axis=n_axis_to_roll)
                rescaled_mask = np.roll(rescaled_mask, amount, axis=n_axis_to_roll)

            Image.fromarray(sample).save(f"{savedir}/synth_imgs/{str(i).zfill(zero_pads)}.png")
            Image.fromarray(rescaled_mask).save(f"{savedir}/masks/{str(i).zfill(zero_pads)}.png")

            if j > n:
                break

