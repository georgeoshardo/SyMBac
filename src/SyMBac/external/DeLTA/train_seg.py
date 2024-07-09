'''
This script trains the cell segmentation U-Net

@author: jblugagne
'''
from model import unet_seg
from data import trainGenerator_seg
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import config as cfg

# Parameters:
batch_size = 10
epochs = 200
steps_per_epoch = 250
patience = 30

#Data generator parameters:
data_gen_args = dict(rotation = 0.5,
                     shiftX=.05,
                     shiftY=.05,
                     zoom=.15,
                     horizontal_flip=True,
                     histogram_voodoo=True,
                     illumination_voodoo=True)

# Generator init:
myGene = trainGenerator_seg(batch_size,
                            os.path.join(cfg.training_set_seg,'img'),
                            os.path.join(cfg.training_set_seg,'seg'),
                            os.path.join(cfg.training_set_seg,'wei'),
                            augment_params = data_gen_args,
                            target_size = cfg.target_size_seg)

# Define model:
model = unet_seg(input_size = cfg.target_size_seg+(1,))
model.summary()

# Callbacks:
model_checkpoint = ModelCheckpoint(cfg.model_file_seg, 
                                    monitor='loss',
                                    verbose=2,
                                    save_best_only=True)
early_stopping = EarlyStopping(monitor='loss',
                               mode='min',
                               verbose=2,
                               patience=patience)

# Train:
history = model.fit_generator(myGene,
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              callbacks=[model_checkpoint, early_stopping])