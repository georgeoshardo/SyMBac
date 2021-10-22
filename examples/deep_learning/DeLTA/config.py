# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:42:05 2021

@author: jblugagne
"""

# If running into OOM issues or having trouble with cuDNN loading, try setting
# memory_growth_limit to a value in MB: (eg 1024, 2048...)
memory_growth_limit = None

if memory_growth_limit is not None:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_growth_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

# Path to model files:
# Download latest models first:
# https://drive.google.com/drive/folders/1nTRVo0rPP9CR9F6WUunVXSXrLNMT_zCP
model_file_chambers = 'C:/DeepLearning/DeLTA_data/mother_machine/models/chambers_id_tessiechamp.hdf5'
model_file_seg = 'C:/DeepLearning/DeLTA_data/mother_machine/models/unet_moma_seg_multisets.hdf5'
model_file_track = 'C:/DeepLearning/DeLTA_data/mother_machine/models/unet_moma_track_multisets.hdf5'

# Model parameters:
target_size_chambers = (512, 512)
target_size_seg = (256, 32)

# Training sets:
training_set_chambers = 'C:/DeepLearning/DeLTA_data/mother_machine/training/chambers_seg_set/train'
training_set_seg = 'C:/DeepLearning/DeLTA_data/mother_machine/training/segmentation_set/train_multisets/'
training_set_track = 'C:/DeepLearning/DeLTA_data/mother_machine/training/tracking_set/train_multisets/'

# Misc:
whole_frame_drift = False # Whether to use the whole frame for drift correction
min_chamber_area = 500 # Minimum size of chambers in pixels for area filtering (0 to +Inf)
min_cell_area = None # Minimum area of cells *IN THE 256x32 SEGMENTATION OUTPUT MASK OF THE UNET*
write_mp4_movies = True # Whether to save MP4 movies showing the DeLTA segmentation & tracking results
rotation_correction = True # Whether to perform automated image rotation correction
