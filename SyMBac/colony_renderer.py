import random
from itertools import cycle, islice
import os
import ray
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

import cupy as cp
from joblib import Parallel, delayed

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
        if "phase" in self.PSF.mode.lower():
        	temp_kernel = gaussian_filter(temp_kernel, 8.7, mode="reflect")

        convolved = convolve_rescale(scene, temp_kernel, 1/self.resize_amount, rescale_int=True)

        if "phase" in self.PSF.mode.lower():
            bg = self.random_perlin_generator()
            convolved += gaussian_filter(np.rot90(bg)/np.random.uniform(1000,3000), np.random.uniform(1,3), mode="reflect")

        convolved = random_noise((convolved), mode="poisson")
        convolved = random_noise((convolved), mode="gaussian", mean=1, var=0.0002, clip=False)

        convolved = rescale_intensity(convolved.astype(np.float32), out_range=(0, np.iinfo(np.uint16).max)).astype(np.uint16)

        return convolved

    def generate_random_samples(self, n, roll_prob, savedir, GPUs = (0,) , n_jobs = 1, batch_size = 20):
        n_GPUs = len(GPUs)
        if n_GPUs > 1:
            n_jobs = n_GPUs
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

        ray.init(num_gpus=n_GPUs)

        @ray.remote(num_gpus=1/n_jobs)
        def run_on_GPU(batch, zero_pads, gpu_id, n_jobs):
            #with cp.cuda.Device(gpu_id):
            s = cp.cuda.Stream(non_blocking = True)
            with s:
                def run_batch(j, i):
                    if j > n:
                        pass
                    else:
                        sample = self.render_scene(i)
                        mask = self.mask_loader(i)
                        rescaled_mask =  rescale(mask, 1 / self.resize_amount, anti_aliasing=False, order=0, preserve_range=True).astype(np.uint16)
    
                        if np.random.rand() < roll_prob:
                            n_axis_to_roll, amount = random.choice([(0, int(sample.shape[0]/2)), (1, int(sample.shape[1]/2)), ([0,1], (int(sample.shape[0]/2), int(sample.shape[1]/2)))])
                            sample = np.roll(sample, amount, axis=n_axis_to_roll)
                            rescaled_mask = np.roll(rescaled_mask, amount, axis=n_axis_to_roll)
    
                        Image.fromarray(sample).save(f"{savedir}/synth_imgs/{str(i).zfill(zero_pads)}.png")
                        Image.fromarray(rescaled_mask).save(f"{savedir}/masks/{str(i).zfill(zero_pads)}.png")
    
                        #if j > n:
                        #    break
            Parallel(n_jobs=1)(delayed(run_batch)(j, i) for j, i in batch)

        def batched(iterable, n):
            "Batch data into tuples of length n. The last batch may be shorter."
            # batched('ABCDEFG', 3) --> ABC DEF G
            if n < 1:
                raise ValueError('n must be at least one')
            it = iter(iterable)
            while (batch := tuple(islice(it, n))):
                yield batch

        n_batches = int(np.ceil(n / batch_size))
        batched_idxs = batched(  islice(enumerate(cycle(range(len(self.OPL_dirs)))), 0, n), batch_size)
        batched_zip = zip(batched_idxs,cycle(GPUs))
        #batched_zip = islice(batched_zip, 0, n_batches)

        #Parallel(n_jobs=n_jobs, backend="loky")(delayed(run_on_GPU)(batch, zero_pads, gpu_id) for batch, gpu_id in tqdm(batched_zip, total=n_batches) )
        ray.get([run_on_GPU.remote(batch, zero_pads, gpu_id, n_jobs) for batch, gpu_id in batched_zip])
        #for j, i in tqdm(enumerate(cycle(range(len(self.OPL_dirs)))), total = n): 
        #    sample = self.render_scene(i)
        #    mask = self.mask_loader(i)
        #    rescaled_mask =  rescale(mask, 1 / self.resize_amount, anti_aliasing=False, order=0, preserve_range=True).astype(np.uint16)#

        #    if np.random.rand() < roll_prob:
        #        n_axis_to_roll, amount = random.choice([(0, int(sample.shape[0]/2)), (1, int(sample.shape[1]/2)), ([0,1], (int(sample.shape[0]/2), int(sample.shape[1]/2)))])
        #        sample = np.roll(sample, amount, axis=n_axis_to_roll)
        #        rescaled_mask = np.roll(rescaled_mask, amount, axis=n_axis_to_roll)#

        #    Image.fromarray(sample).save(f"{savedir}/synth_imgs/{str(i).zfill(zero_pads)}.png")
        #    Image.fromarray(rescaled_mask).save(f"{savedir}/masks/{str(i).zfill(zero_pads)}.png")

        #    if j > n:
        #        break

