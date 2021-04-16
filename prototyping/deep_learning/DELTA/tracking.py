'''
This script runs the tracking U-Net on data out of the segmentation U-Net.

Images are processed in batches of 512, although the number of actual samples
run through the tracking U-Net will depend on the number of cells in each 
image.

@author: jblugagne
'''
from data import saveResult_track, predictCompilefromseg_track
from model import unet_track
from os import listdir

# Files:
DeLTA_data = 'C:/DeepLearning/DeLTA_data/'
images_folder = DeLTA_data + 'mother_machine/evaluation/preprocessed/img/'
segmentation_folder = DeLTA_data + 'mother_machine/evaluation/seg_output/'
outputs_folder = DeLTA_data + 'mother_machine/evaluation/track_output/'
model_file = DeLTA_data + 'mother_machine/models/unet_moma_track_multisets.hdf5'
unprocessed = listdir(images_folder)

# Parameters:
target_size = (256, 32)
input_size = target_size + (4,)
process_size = 512 # This is the number of frames to consider _BEFORE_ each sample has been subdivided into cell-specific tracking samples

# Load up model:
model = unet_track(input_size = input_size)
model.load_weights(model_file)

# Process
while(unprocessed):
    # Pop out filenames
    ps = min(process_size,len(unprocessed))
    to_process = unprocessed[0:ps]
    del unprocessed[0:ps]
    
    # Get data:
    inputs, seg_filenames = predictCompilefromseg_track(images_folder, segmentation_folder,
                                                        files_list = to_process, 
                                                        target_size = target_size)
    
    # Predict:
    results = model.predict(inputs,verbose=1)
    
    # Save (use the filenames list from the data compiler)
    saveResult_track(outputs_folder,results, files_list = seg_filenames)