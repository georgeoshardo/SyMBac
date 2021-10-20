#!/usr/bin/env bash
conda create --name SyMBac python=3.10 -y
conda activate SyMBac
conda install -n SyMBac -c conda-forge tifffile scikit-image matplotlib tqdm pandas natsort jupyterlab ipywidgets joblib -y
jupyter nbextension enable --py widgetsnbextension

pip install tensorflow csbdeep stardist elasticdeform conda-forge tifffile scikit-image matplotlib tqdm pandas natsort jupyterlab ipywidgets joblib "napari[all]"
pip install "napari[all]"
