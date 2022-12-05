# SyMBac: Synthetic Micrographs of Bacteria

[Read the paper: Synthetic Micrographs of Bacteria (SyMBac) allows accurate segmentation of bacterial cells using deep neural networks
](https://doi.org/10.1186/s12915-022-01453-6
), Georgeos Hardo, Maximillian Noka, Somenath Bakshi

[***NEW: Read the docs***](https://symbac.readthedocs.io/en/latest/)

<img src="https://github.com/georgeoshardo/SyMBac/raw/main/readme_files/symbac_sliders.gif" alt="drawing" width="600" height="400"/>

  * [What is it?](#what-is-it-)
  * [Why would I want to generate synthetic images?](#why-would-i-want-to-generate-synthetic-images-)
  * [How do I use these synthetic images?](#how-do-i-use-these-synthetic-images-)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
    + [If you're using a GPU:](#if-you-re-using-a-gpu-)
  * [Usage](#usage)
  * [FAQs](#faqs)



## What is it?

SyMBac is a tool to generate synthetic phase contrast or fluorescence images of bacteria. Currently the tool only supports bacteria growing in the mother machine, however support for bacteria growing in monolayers (and maybe even biofilms!) is coming. 

<img src="https://github.com/georgeoshardo/SyMBac/raw/main/readme_files/example_comparison.jpeg" alt="comparisons" width="600" />



## Why would I want to generate synthetic images?

Because you're sick of generating your own training data by hand! Synthetic images provide an instant source of high quality and unlimited training data for machine learning image segmentation algorithms! 

The images are tuned to perfectly replicate your experimental setup, no matter what your microscope's objective is (we have tested 20x air all the way to 100x oil), no matter your imaging modality (phase contrast/fluorescence), and no matter the geometry of your microfluidic device. 

Additionally,

* SyMBac is very fast compared to humans:

<img src="https://github.com/georgeoshardo/SyMBac/raw/main/readme_files/speed_comparison.png" alt="comparisons" width="400"  />

* The image generation process uses a rigid body physics model to simulate bacterial growth, 3D cell geometry to calculate the light's optical path, and a model of the phase contrast/fluorescence optics (point spread function), with some post-rendering optimisation to match image similarity:

<img src="https://github.com/georgeoshardo/SyMBac/raw/main/readme_files/image_generation.jpeg" alt="comparisons"  width="600" />

## How do I use these synthetic images?

That is up to you. SyMBac is **not** a machine learning tool. It is a tool to generate unlimited free training data which accurately represents your experiment. It is up to you to train a machine learning network on these synthetic images. We do however provide example notebooks for how to train a U-net (as implemented by [DeLTA](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007673)).

 <img src="https://github.com/georgeoshardo/SyMBac/raw/main/readme_files/training.png" alt="comparisons"  width="350" />

## Prerequisites

Please make sure you have an NVIDIA GPU and a working installation of `CUDA` and `cudNN`. If you don't have an NVIDIA GPU then the convolution will default to the CPU, and be **very** slow.

SyMBac is meant to be run *interactively* (in a notebook + with a small Qt/GTK intWWerface), so make sure that you are running this on a local machine (you should have access to the machine's display).

## Installation

```sh
pip install SyMBac
```

Or to install the development version, run:

```sh
pip install git+https://github.com/georgeoshardo/SyMBac
```

Activate the Jupyter widgets extension. This is needed to interact with slides in the notebooks to optimise images. 

```sh
jupyter nbextension enable --py widgetsnbextension
```

### If you're using a GPU:

Check the version of `CUDA` you have installed using `nvcc --version` and install the appropriate version of [cupy](https://cupy.dev/). For example, if you have `CUDA 11.4` you would install as follows:

```sh
pip install cupy-cuda114
```

If you installed CUDA on Ubuntu 18.04+ using the new Nvidia supplied repositories, it is a real possibility that `nvcc` won't work. Instead check your CUDA version using `nvidia-smi`.

### If you aren't using a GPU:

See FAQs "Do I need to have a GPU?"

## Usage

[***Read the docs***](https://symbac.readthedocs.io/en/latest/)

## FAQs

* Do I need to have a GPU?
  * No, although image synthesis will be around 40x slower on the CPU. SyMBac will detect that you do not have CuPy installed and default to using CPU convolution.
  * Interactive image optimisation will be very painful on the CPU. By default I turn off slider interactivity if you are using the CPU, so that you can move a slider without the CPU being maxed out. This means that every time you move a slider you must click the button to update the image (do a convolution).
* Can I generate fluorescence images as well?
  * Yes, you can do fluorescence image generation, just make sure that in the interactive image generation part of the code, you select fluorescence.
  * Since our fluorescence kernel is defined to be a subset of the phase contrast kernel, you can choose **any** condenser, and your fluorescence kernel should be correct. Just ensure that the imaging wavelength, numerical aperture, refractive index, and pixel size are set correctly.
* What format do my images need to be in?
  * The real images you are trying to replicate should be in the format of single-trench timeseries images. If you are unsure what this is, you can call `get_sample_images()["E. coli 100x"]` from `SyMBac.misc`for an example image.
* I'm getting libGL MESA-LOADER errors
  * See https://unix.stackexchange.com/questions/655495/trying-to-run-pygame-on-my-conda-environment-on-my-fresh-manjaro-install-and-ge