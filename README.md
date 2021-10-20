# SyMBac: Synthetic Micrographs of Bacteria

## What is it?

SyMBac is a tool to generate synthetic phase contrast or fluorescence images of bacteria. Currently the tool only supports bacteria growing in the mother machine, however support for bacteria growing in monolayers (and even biofilms!) is coming. 

## Why would I want to generate synthetic images?

Because you're sick of generating your own training data by hand! Synthetic images provide an instant source of high quality and unlimited training data for machine learning image segmentation algorithms! 

The images are tuned to perfectly replicate your experimental setup, no matter what your microscope's objective is (we have tested 20x air all the way to 100x oil), no matter your imaging modality (phase contrast/fluorescence), and no matter the geometry of your microfluidic device. 

## How do I use these synthetic images?

That is up to you. SyMBac is **not** a machine learning tool. It is a tool to generate unlimited free training data which accurately represents your experiment. It is up to you to train a machine learning network on these synthetic images. We do however provide example notebooks for how to train a U-net (as implemented by [DeLTA](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007673)). 

## Prerequisites

Please make sure you an NVIDIA GPU and a working installation of `cuda` and `cudNN` 

SyMBac is meant to be run *interactively* (in a notebook + with a small Qt/GTK interface), so make sure that you are running this on a local machine (you should have access to the machine's display).

## Installation

Using Anaconda, create an environment and enter it.

```sh
conda create --name SyMBac python=3.9 -y
conda activate SyMBac
```

Install all the required packages in `requirements.txt`

```sh
pip install -r requirements.txt
```

Activate the Jupyter widgets extension. This is needed to interact with slides in the notebooks to optimise images. 

```sh
jupyter nbextension enable --py widgetsnbextension
```



## Usage

Please bear with me while I upload installation documentation!
Please note that the previous name of this project was SYMPTOMM, and so I am currently in the process of renaming these references in the library.
