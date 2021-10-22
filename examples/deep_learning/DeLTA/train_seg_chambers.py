'''
This script trains the chambers segmentation U-Net.

@author: jblugagne
'''
from model import unet_chambers
from data import trainGenerator_seg
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import config as cfg

# Parameters:
batch_size = 1
epochs = 100
steps_per_epoch = 250
patience = 15

#Data generator parameters:
data_gen_args = dict(rotation = 3,
                     shiftX=.1,
                     shiftY=.1,
                     zoom=.25,
                     horizontal_flip=True,
                     vertical_flip=True,
                     rotations_90d=True,
                     histogram_voodoo=True,
                     illumination_voodoo=True)

# Generator init:
myGene = trainGenerator_seg(batch_size,
                            os.path.join(cfg.training_set_chambers,'img'),
                            os.path.join(cfg.training_set_chambers,'seg'),
                            None,
                            augment_params = data_gen_args,
                            target_size = cfg.target_size_chambers)

# Define model:
model = unet_chambers(input_size = cfg.target_size_chambers+(1,))
model.summary()

# Callbacks:
model_checkpoint = ModelCheckpoint(cfg.model_file_chambers,
                                   monitor='loss',
                                   verbose=1,
                                   save_best_only=True)
early_stopping = EarlyStopping(monitor='loss',
                               mode='min',
                               verbose=1,
                               patience=patience)

# Train:
history = model.fit_generator(myGene,
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              callbacks=[model_checkpoint, early_stopping])