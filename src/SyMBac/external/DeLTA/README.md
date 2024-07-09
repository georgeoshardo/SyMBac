# DeLTA

DeLTA (Deep Learning for Time-lapse Analysis) is a deep learning-based image processing pipeline for segmenting and tracking single cells in time-lapse microscopy movies.
![](https://gitlab.com/dunloplab/delta/-/raw/images/DeLTAexample.gif)

See/Cite out paper in PLOS Computational Biology:

[Lugagne, J.-B., Lin, H., & Dunlop, M. J. (2020). DeLTA: Automated cell segmentation, tracking, and lineage reconstruction using deep learning. PLOS Computational Biology, 16(4), e1007673](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007673).

##### Overview
Our pipeline is centered around two [U-Net](https://arxiv.org/abs/1505.04597) neural networks that are used sequentially:
1. To perform semantic binary segmentation of our cells as in the original U-Net paper.
2. To track cells from one movie frame to the next, and to identify cell divisions and mother/daughter cells.

In our application, we focus on *E. coli* cells trapped in a microfluidic device known as a ["mother machine"](https://www.cell.com/current-biology/fulltext/S0960-9822%2810%2900524-5). However, we have been and are still working on variants of our work for segmentation and tracking of yeast and rod-shaped bacteria growing in agar pads or 'biopixel'-like microfluidic devices. See the yeast branch of this repository.

The U-Nets are implemented in Tensorflow 2.3 via the Keras API. In recent months we have been moving towards implementing the entire software in python, but we still provide a suite of scripts and GUIs in Matlab to interface DeLTA with Matlab and help users create their own training sets. See our [DeLTA interfacing](https://gitlab.com/dunloplab/delta-interfacing) repository for that.

We try to comment/document and make the code accessible and easily adaptable. We provide example datasets to illustrate how the code works (see below).

##### Additions since publication
Since submission of the manuscript to PLOS Comp Biol, we have kept working on this repository and a few new features are not described in the paper. These include:

- A third U-Net to detect mother machine chambers in microscopy images. This allows us crop out chamber images to then feed into the segmentation and tracking pipeline. Object detection models, such as YOLO or RCNN would have been more appropriate for this task, but this works for us and it keeps the code simple. Models and training sets are provided in the new archive.

- A pipeline that runs completely in python. The script in [pipeline.py](pipeline.py) loads images from a tif file, bio-formats compatible file (nd2, czi, oib...), or a sequence of images in a folder. It then identifies chambers, segments cells, tracks them, reconstructs the lineage, and extracts features such as cell length and fluorescence. This greatly speeds up processing speed, as it removes a lot of manual steps and input-output operations. It is however not as simple as the [segmentation.py](tracking.py) and [tracking.py](tracking.py) scripts.

- A script and an environment to process Bio-Formats compatible files (nd2, czi, oib...) into a sequence of single tif files. As of March 2021 this is obsolete as the python-bioformats library has been ported to python 3 and our pipeline script can now read such files directly. See Bio-formats section below

Note that, to minimize modifications to the old code, we kept data manipulation functions related to those new features in the new [utilities.py](utilities.py) file. This creates a lot of redundant functions, but it ensures that we do not get unexpected new bugs for the parts of the code that worked fine before. Also, the new part of the code uses the openCV library for performance purposes, while older parts of the code use scikit-image for its relative simplicity.

## Install
We are using this code on several different Linux and Windows systems, and unless otherwise specified the instructions apply for both.

### GPU Requirements
While this code can run on CPU, we recommend running it on GPU for significantly faster processing. To do so you will need [an nvidia gpu with a cuda compute capability of 3.5 or higher](https://developer.nvidia.com/cuda-gpus) and [an up-to-date gpu driver](https://www.nvidia.com/Download/index.aspx?lang=en-us)). If no compatible GPU/driver is found on the system, tensorflow will run on CPU.

For more information see [here](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/) and [here](https://www.tensorflow.org/guide/gpu).

### Anaconda installation
We recommend installing necessary modules and libraries with [Anaconda](https://www.anaconda.com/distribution/). Anaconda is easy to install, and will allow you to setup environments that are not in conflict with each other and will streamline the installation process.

Note: On Windows, you might need to first [install c++ support with Visual Studio](https://docs.microsoft.com/en-us/cpp/build/vscpp-step-0-installation).

From inside the DeLTA folder, the following commands will install and activate the environment:

    (base)$ conda env create -f delta_env.yml
	(base)$ conda activate delta_env
Note that the library versions listed in the [delta_env.yml](delta_env.yml) file have been tested and work on Linux and Windows, but more recent versions should work as well. We use the Spyder IDE, but it can be removed from the environment file.

## Example datasets
You can create your own datasets for your setup using our [suite of Matlab scripts and GUIs](https://gitlab.com/dunloplab/delta-interfacing), however we recommend you check out how data is formatted in our example datasets first:
[DeLTA_data (google drive, 19GB zip file)](https://drive.google.com/drive/u/0/folders/1nTRVo0rPP9CR9F6WUunVXSXrLNMT_zCP) [~38 GB unzipped]

Once unzipped, you should update the DeLTA_data folder path at the beginning of all scripts to test them.

In that same google folder you can also find our latest trained model files. Get in touch if you want us to take a look at your data and add it to our training sets.

## Example run:

The original workflow of DeLTA was:

**Training**

0. Design new training sets for segmentation and tracking (see [DeLTA interfacing](https://gitlab.com/dunloplab/delta-interfacing))
1. Train the segmentation and tracking U-Net on the training sets

**Prediction**

2. Run trained segmentation U-Net on new data
3. Run trained tracking U-Net on output of segmentation U-Net
4. Run post-processing on tracking output to compile into a user-friendly format (see [DeLTA interfacing](https://gitlab.com/dunloplab/delta-interfacing))

Now, after training, prediction can be run with only one script: [pipeline.py](pipeline.py). Note that there is an area filtering parameter for the chambers (min_chamber_area) in the script that is set manually. Set it to 0 or None if you don't want to apply this area filtering step.

![](https://gitlab.com/dunloplab/delta/-/raw/images/DeLTA_code.svg)

Note that we provide trained models and you can run the prediction steps without having to retrain on our DeLTA_data training sets.




### Training
You can test training with the datasets we provide with the [train_seg.py](train_seg.py) script for segmentation, the [train_track.py](train_track.py) script for tracking, and the [train_seg_chambers.py](train_seg_chambers.py) script for chamber identification. You can run them from within an IDE like spyder or from the shell.

### Prediction
Once the models are trained, you can run predictions on larger datasets using the [segmentation.py](segmentation.py) and [tracking.py](tracking.py) scripts sequentially if chamber images are already cropped out. (see mother_machine/evaluation/preprocessed/img folder in the datasets archive).

Otherwise, the [pipeline.py](pipeline.py) script can process an entire experiment from a bioformats-compatible file or a sequence of image files saved in the same folder. You can test the pipeline on the evaluation .nd2 movie provided in the example dataset. A folder will be created next to the experiment file or within its folder. MAT files containing segmentation and tracking data for each position will be saved to it. The structure of the output MAT files is described in [data_structure.txt](data_structure.txt). If `write_mp4_movies` in [config.py](config.py) is set to True, the pipeline will also generate mp4 movie files for each movie illustrating the segmentation and tracking performance of DeLTA for each position.

## Bio-Formats microscopy files
Note: Until recently, the python-bioformats library that allow us to read most [microscopy file formats](https://docs.openmicroscopy.org/bio-formats/5.7.1/supported-formats.html "Bio-Formats compatible") like .nd2, .czi, .oif etc did not work with python 3. We provided the following tools to export those formats form a python 2 environment to a sequence of images that could then be used as input for our pipeline. With the recent port of this library to python 3, these instructions and scripts have become obsolete and they will be removed in a future version.

1. First, you can convert various microscopy formats into ome-tiff with Bio-Formats' [bfconvert](https://docs.openmicroscopy.org/bio-formats/5.7.1/users/comlinetools/conversion.html "bfconvert") tool. Our xpreader object can read OME-TIFF files converted this way, although it can run into memory issues if the file is too big.

2. We also provide a small python 2 script, [bioformats2sequence.py](bioformats2sequence.py) that can run in the conda environment described in [bioformats_env.yml](bioformats_env.yml). This script writes all frames from a bioformats-compatible file to single tiff files sequence compatible with our pipeline script. We have only tested this with nd2 and czi files, although other formats should also work.

### Bio-formats Setup

On Windows you will need to install a recent [Java JDK](https://www.oracle.com/java/technologies/javase-downloads.html) (≥8), [.NET support for Visual studio](https://docs.microsoft.com/en-us/dotnet/framework/install/guide-for-developers), and the [Visual C++ compiler for python 2.7](https://aka.ms/vcpython27). On Linux you *should* be good to go. Then run the following commands:

    (base)$ conda env create -f bioformats_env.yml
    (base)$ conda activate bioformats_env
    (bioformats_env)$ python bioformats2sequence.py /path/to/microscopyfile.nd2 /path/to/outputfolder/

The path to the output folder can then be passed to the xpreader object in [pipeline.py](pipeline.py) (after activating delta_env)

Note: On more recent versions of spyder, you may get an encoding error on startup. Try `$env:PYTHONIOENCODING = "UTF-8"` (Windows) or `set PYTHONIOENCODING=UTF-8` (Unix). This seems to not be an issue if you launch from an anaconda Powershell prompt.

## Troubleshooting
### OOM - Out of memory (GPU)
If you run the code on your GPU, you might run into cache memory problems. This is both linked to the batch size and the size of the images you are using. The batch size is fairly straightforward to change, just lower the value at the beginning of the [train_seg.py](train_seg.py) or [train_track.py](train_track.py) files. Note that lower batch sizes may mean slower training convergence or lower performance overall. You might want to increase the number of epochs or steps per epoch to compensate.

The other solution would be to use a smaller image target size. However if the original training images and masks are for example 512×512, downsizing them to 256×256 will reduce the memory footprint, but it might cause some borders between cells in the binary masks to disappear. Instead, you would have to modify your training images size upstream of DeLTA to make sure that your training set does feature cell-to-cell borders in the segmentation masks.

### cuDNN (or other libraries) not loading
We have run into OOM errors or some GPU-related libraries failing to load or initialize on laptops, and the problems seems to come from tensorflow using too much memory on the GPU cache. See the "Limiting GPU memory growth" section on [this tensorflow help page](https://www.tensorflow.org/guide/gpu). You can try setting the `memory_growth_limit` parameter at the beginning of [config.py](config.py) to a set value in MB (eg 1024, 2048...).

### Prefix already exists
If the installation of one of the environments was interrupted for one reason or another, you might need to remove it before attempting to install it again:

    conda env remove -n delta_env (or bioformats_env)
    conda clean --all

### max() arg is an empty sequence / Some chambers are not analyzed
In [pipeline.py](pipeline.py), there is an area filtering parameter `min_chamber_area` that you can lower or set to 0 or None.
Bear in mind that this filtering operation typically removes chambers that are on the edges of the image.

### NaN values in fluorescence, cell length, area etc... time-series after processing
This should only happen if a cell is shrunk out of existence after the resizing step that transforms label_stack into label_stack_resized (see [data_structure.txt](data_structure.txt)). This typically happens in chambers that are right on the edge of the FOV and that are only partially visible, or when a focussing error causes the segmentation U-Net to identify very small cells. You can try to filter them out with the `min_cell_area` parameter in [config.py](config.py) or  simply ignore NaN values.