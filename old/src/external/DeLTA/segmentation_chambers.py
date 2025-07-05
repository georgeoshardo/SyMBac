"""
This script will run the chambers identification/segmentation U-Net.

To see how to extract chamber images with this segmentation masks, see the
preprocessing section/cell of pipeline.py and getChamberboxes() in utilities.py

@author: jblugagne
"""

from os import listdir

from data import postprocess, predictGenerator_seg, saveResult_seg
from model import unet_chambers

# Files:
DeLTA_data = "C:/DeepLearning/DeLTA_data/"
inputs_folder = (
    DeLTA_data + "mother_machine/evaluation/sequence/"
)  # run bioformats2sequence.py first
outputs_folder = DeLTA_data + "mother_machine/evaluation/chambers_masks/"
model_file = DeLTA_data + "mother_machine/models/chambers_id_tessiechamp.hdf5"
input_files = listdir(inputs_folder)


# Parameters:
target_size = (512, 512)
input_size = target_size + (1,)

# Load up model:
model = unet_chambers(input_size=input_size)
model.load_weights(model_file)


# Predict:
predGene = predictGenerator_seg(inputs_folder, files_list=input_files, target_size=target_size)
results = model.predict_generator(predGene, len(input_files), verbose=1)

# Post process results:
results[:, :, :, 0] = postprocess(results[:, :, :, 0])

# Save to disk:
saveResult_seg(outputs_folder, results, files_list=input_files)
