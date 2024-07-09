from keras import layers
from keras import activations
from keras import ops
import keras

def ESPCN_block(inputs, upscale_factor=4, channels=1):
    conv_args = {
        "kernel_initializer": "he_normal",
        "padding": "same",
    }
    
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.BatchNormalization()(x)
    x = activations.relu(x)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.BatchNormalization()(x)
    x = activations.relu(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.BatchNormalization()(x)
    x = activations.relu(x)
    x = layers.Conv2D(channels * (upscale_factor**2), 3, **conv_args)(x)
    x = layers.BatchNormalization()(x)
    x = activations.relu(x)
    SR_outputs = DepthToSpace(upscale_factor, name="sr_outputs")(x)

    return SR_outputs

class DepthToSpace(layers.Layer):
    def __init__(self, block_size,name):  # Accept arbitrary keyword arguments
        super().__init__()  # Pass these arguments to the superclass initializer
        self.block_size = block_size
        self.name = name
        
    def call(self, input):
        batch, height, width, depth = ops.shape(input)
        depth = depth // (self.block_size**2)

        x = ops.reshape(
            input, [batch, height, width, self.block_size, self.block_size, depth]
        )
        x = ops.transpose(x, [0, 1, 3, 2, 4, 5])
        x = ops.reshape(
            x, [batch, height * self.block_size, width * self.block_size, depth]
        )
        return x


def Unet_block(inputs):

    def conv_down_encode(x, filters, kernel_size = 3, padding = "same", kernel_initializer = "he_normal", activation = "relu", batch_norm = True, dropout = False):
        x = conv_block(x, filters = filters, kernel_size = kernel_size, padding = padding, kernel_initializer = kernel_initializer, activation = activation, batch_norm = batch_norm)

        if dropout:
            x = layers.Dropout(dropout)(x)

        p = layers.MaxPooling2D(pool_size=(2, 2))(x)
        return p, x
    
    def conv_block(x, filters, kernel_size = 3, padding = "same", kernel_initializer = "he_normal", activation = "relu", batch_norm = True):
        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)

        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return x
    
    def conv_up_decode(x, concat_layer, filters, kernel_size = 3, padding = "same", kernel_initializer = "he_normal", activation = "relu", batch_norm = True, dropout = False):
        #x = layers.Conv2DTranspose(filters, kernel_size = 2, strides = 2, padding=padding)(x)
        x = layers.UpSampling2D(size=(2,2))(x)
        print(x.shape)
        print(concat_layer.shape)
        x = layers.Conv2D(filters, kernel_size = 2, padding=padding, kernel_initializer=kernel_initializer)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Concatenate()([concat_layer,x])
        x = conv_block(x, filters = filters, kernel_size = kernel_size, padding = padding, kernel_initializer = kernel_initializer, activation = activation, batch_norm = batch_norm)
        if dropout:
            x = layers.Dropout(dropout)(x)
        return x

    print(inputs.shape)
    p1, c1 = conv_down_encode(inputs, filters = 64)    
    p2, c2 = conv_down_encode(p1, filters = 128)
    p3, c3 = conv_down_encode(p2, filters = 256)
    p4, c4 = conv_down_encode(p3, filters = 512, dropout = 0.5)

    b1 = conv_block(p4, filters = 1024)  # Bottom block

    u4 = conv_up_decode(b1, c4, filters = 512)
    u3 = conv_up_decode(u4, c3, filters = 256)
    u2 = conv_up_decode(u3, c2, filters = 128)
    u1 = conv_up_decode(u2, c1, filters = 64)

    mask_outputs = layers.Conv2D(1, 1, activation="sigmoid", name="mask_outputs")(u1)

    return mask_outputs

def SR_seg(input_shape, upscale_factor = 4, channels = 1):
    inputs = keras.Input(shape=(input_shape[0], input_shape[1], channels))

    SR_outputs = ESPCN_block(inputs, upscale_factor = upscale_factor, channels = channels)
    mask_outputs = Unet_block(SR_outputs)

    return keras.Model(inputs, [SR_outputs, mask_outputs])