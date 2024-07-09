'''
This script trains the tracking U-Net.

@author: jblugagne
'''
from model import unet_track
from data import trainGenerator_track
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import config as cfg

# Parameters:
class_weights = (1,1,1) # same weight for all classes
batch_size = 5
epochs = 400
steps_per_epoch = 250
patience = 30

#Data generator parameters:
data_gen_args = dict(rotation = 1,
                     shiftX=.05,
                     shiftY=.05,
                     zoom=.15,
                     horizontal_flip=True,
                     histogram_voodoo=True,
                     illumination_voodoo=True)

# Generator init:
myGene = trainGenerator_track(batch_size,
                              os.path.join(cfg.training_set_track,'img'),
                              os.path.join(cfg.training_set_track,'seg'),
                              os.path.join(cfg.training_set_track,'previmg'),
                              os.path.join(cfg.training_set_track,'segall'),
                              os.path.join(cfg.training_set_track,'mother'),
                              os.path.join(cfg.training_set_track,'daughter'),
                              data_gen_args,
                              target_size = cfg.target_size_seg)

# Define model:
model = unet_track(input_size = cfg.target_size_seg+(4,),
                   class_weights = class_weights)
model.summary()

# Callbacks:
model_checkpoint = ModelCheckpoint(cfg.model_file_track,
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