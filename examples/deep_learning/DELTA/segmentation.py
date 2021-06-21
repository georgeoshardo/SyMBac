'''
This script runs the segmentation U-Net on data that has been pre-processed to
crop out chamber images. See delta-interfacing/preprocessing.m or pipeline.py.

The images are processed by batches of 4096 to prevent memory issues.

@author: jblugagne
'''
from data import saveResult_seg, predictGenerator_seg, postprocess
from model import unet_seg
from os import listdir

# Files:
DeLTA_data = 'C:/DeepLearning/DeLTA_data/'
inputs_folder = DeLTA_data + 'mother_machine/evaluation/preprocessed/img/'
outputs_folder = DeLTA_data + 'mother_machine/evaluation/seg_output/'
model_file = DeLTA_data + 'mother_machine/models/unet_moma_seg_multisets.hdf5'
unprocessed = listdir(inputs_folder)

# Parameters:
target_size = (256, 32)
input_size = target_size + (1,)
process_size = 4096

# Load up model:
model = unet_seg(input_size = input_size)
model.load_weights(model_file)

# Process
while(unprocessed):
    # Pop out filenames
    ps = min(process_size,len(unprocessed))
    to_process = unprocessed[0:ps]
    del unprocessed[0:ps]
    
    # Predict:
    predGene = predictGenerator_seg(inputs_folder, files_list = to_process, target_size = target_size)
    results = model.predict_generator(predGene,len(to_process),verbose=1)
    
    # Post process results:
    results[:,:,:,0] = postprocess(results[:,:,:,0])
    
    # Save to disk:
    saveResult_seg(outputs_folder,results, files_list = to_process)