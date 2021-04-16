'''
This script trains the cell segmentation U-Net

@author: jblugagne
'''
# pip install "scikit_image==0.16.2"
from model import unet_seg
from data import trainGenerator_seg
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
# Files:
DeLTA_data =  "/home/georgeos/Storage/Dropbox (Cambridge University)/PhD_Georgeos_Hardo/ML_based_segmentation_results/40x_Ph2_test_1.5/training_data/preprocessed/"
#'/home/georgeos/Documents/MMSynth_data/FalsePosNeg_HADA/training_data_lower_contrast_3/preprocessed/'
training_set = DeLTA_data
model_file = DeLTA_data + '/saved-model-nosmall-{epoch:02d}.hdf5' #'/unet.hdf5'

# Parameters:
target_size = (448, 32)
input_size = target_size + (1,)
batch_size = 30
epochs = 200
steps_per_epoch = 30

#Data generator:
data_gen_args = dict(
                    rotation = 0.5,
                    shiftX=.05,
                    shiftY=.05,
                    zoom=.15,
                    horizontal_flip=True,
                    histogram_voodoo=True,
                    illumination_voodoo=True)

myGene = trainGenerator_seg(batch_size,
                           training_set + 'CROPPED_FILTERED/',
                           training_set + 'CROPPED_MASKS/',
                           training_set + 'WEIGHTMAPS/',
                           augment_params = data_gen_args,
                           target_size = target_size)


# Define model:
model = unet_seg(input_size = input_size)
model.summary()
model_checkpoint = ModelCheckpoint(model_file, monitor='loss',verbose=1, save_best_only=False)

#filepath = "saved-model-{epoch:02d}-{loss:.2f}.hdf5"
# Train it:
model.fit_generator(myGene,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[model_checkpoint])