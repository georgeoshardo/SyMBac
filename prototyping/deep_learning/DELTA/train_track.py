'''
This script trains the tracking U-Net.

@author: jblugagne
'''
from model import unet_track
from data import trainGenerator_track
from tensorflow.keras.callbacks import ModelCheckpoint

# Files:
DeLTA_data = 'C:/DeepLearning/DeLTA_data/'
training_set = DeLTA_data + 'mother_machine/training/tracking_set/train_multisets/'
model_file = DeLTA_data + 'mother_machine/models/unet_moma_track_multisets.hdf5'

# Parameters:
target_size = (256, 32)
input_size = target_size + (4,)
batch_size = 5
epochs = 400
steps_per_epoch = 250

#Data generator:
data_gen_args = dict(
                    rotation = 1,
                    shiftX=.05,
                    shiftY=.05,
                    zoom=.15,
                    horizontal_flip=True,
                    histogram_voodoo=True,
                    illumination_voodoo=True)

myGene = trainGenerator_track(batch_size,
                             training_set + 'img/',
                             training_set + 'seg/',
                             training_set + 'previmg/',
                             training_set + 'segall/',
                             training_set + 'mother/',
                             training_set + 'daughter/',
                             data_gen_args,
                             target_size = target_size)

class_weights = (1,1,1)

# Define model & train it:
model = unet_track(input_size = input_size, class_weights = class_weights)
model.summary()
model_checkpoint = ModelCheckpoint(model_file, monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[model_checkpoint])