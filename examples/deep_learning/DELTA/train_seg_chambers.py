'''
This script trains the chambers segmentation U-Net.

@author: jblugagne
'''
from model import unet_chambers
from data import trainGenerator_seg
from tensorflow.keras.callbacks import ModelCheckpoint

# Files:
DeLTA_data = 'C:/DeepLearning/DeLTA_data/'
training_set = DeLTA_data + 'mother_machine/training/chambers_seg_set/train/'
model_file = DeLTA_data + 'mother_machine/models/chambers_id_tessiechamp.hdf5'

# Parameters:
target_size = (512, 512)
input_size = target_size + (1,)
batch_size = 1
epochs = 100
steps_per_epoch = 250

#Data generator:
data_gen_args = dict(
                    rotation = 3,
                    shiftX=.1,
                    shiftY=.1,
                    zoom=.25,
                    horizontal_flip=True,
                    vertical_flip=True,
                    rotations_90d=True,
                    histogram_voodoo=True,
                    illumination_voodoo=True
                    )


myGene = trainGenerator_seg(batch_size,
                           training_set + 'img/',
                           training_set + 'seg/',
                           None,
                           augment_params = data_gen_args,
                           target_size = target_size)


# Define model:
model = unet_chambers(input_size = input_size)
model.summary()
model_checkpoint = ModelCheckpoint(model_file, monitor='loss',verbose=1, save_best_only=True)


# Train it:
model.fit_generator(myGene,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[model_checkpoint])