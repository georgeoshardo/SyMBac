'''
This file contains model definitions and loss/metrics functions definitions

@author: jblugagne
'''
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam



#%% Losses/metrics
def pixelwise_weighted_binary_crossentropy(y_true, y_pred):
    '''
    Pixel-wise weighted binary cross-entropy loss.
    The code is adapted from the Keras TF backend.
    (see their github)
    
    Parameters
    ----------
    y_true : Tensor
        Stack of groundtruth segmentation masks + weight maps.
    y_pred : Tensor
        Predicted segmentation masks.

    Returns
    -------
    Tensor
        Pixel-wise weight binary cross-entropy between inputs.

    '''
    
    try:
        # The weights are passed as part of the y_true tensor:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass

    epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
    cond = (y_pred >= zeros)
    relu_logits = math_ops.select(cond, y_pred, zeros)
    neg_abs_logits = math_ops.select(cond, -y_pred, y_pred)
    entropy = math_ops.add(relu_logits - y_pred * seg, math_ops.log1p(math_ops.exp(neg_abs_logits)), name=None)
    
    # This is essentially the only part that is different from the Keras code:
    return K.mean(math_ops.multiply(weight, entropy), axis=-1)


def class_weighted_categorical_crossentropy(class_weights):
    '''
    Generate class-weighted categorical cross-entropy loss function.
    The code is adapted from the Keras TF backend.
    (see their github)
    
    Parameters
    ----------
    class_weights : tuple/list of floats
        Weights for each class/category.

    Returns
    -------
    function.
        Class-weighted categorical cross-entropy loss function.

    '''
    
    def loss_function(y_true, y_pred):
        
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, -1, True)
        # manual computation of crossentropy
        epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Multiply each class by its weight:
        classes_list = tf.unstack(y_true * tf.math.log(y_pred), axis=-1)
        for i in range(len(classes_list)):
            classes_list[i] = tf.scalar_mul(class_weights[i], classes_list[i])
        
        # Return weighted sum:
        return - tf.reduce_sum(tf.stack(classes_list, axis=-1), -1)
    
    return loss_function


def unstack_acc(y_true, y_pred):
    '''
    Unstacks the mask from the weights in the output tensor for
    segmentation and computes binary accuracy

    Parameters
    ----------
    y_true : Tensor
        Stack of groundtruth segmentation masks + weight maps.
    y_pred : Tensor
        Predicted segmentation masks.

    Returns
    -------
    Tensor
        Binary prediction accuracy.

    '''
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass    
    
    return keras.metrics.binary_accuracy(seg,y_pred)



#%% Models
# Generic unet declaration:
def unet(input_size = (256,32,1), final_activation = 'sigmoid', output_classes = 1):
    '''
    Generic U-Net declaration.

    Parameters
    ----------
    input_size : tuple of 3 ints, optional
        Dimensions of the input tensor, excluding batch size. 
        The default is (256,32,1).
    final_activation : string or function, optional
        Activation function for the final 2D convolutional layer. see 
        keras.activations
        The default is 'sigmoid'.
    output_classes : int, optional
        Number of output classes, ie dimensionality of the output space of the
        last 2D convolutional layer.
        The default is 1.

    Returns
    -------
    model : Model
        Defined U-Net model (not compiled yet).

    '''
    
    inputs = Input(input_size,  name='true_input')	
	
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = Concatenate(axis = 3)([drop4,up6]) 
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = Concatenate(axis = 3)([conv3,up7])
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = Concatenate(axis = 3)([conv2,up8])
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = Concatenate(axis = 3)([conv1,up9])
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(output_classes, 1, activation = final_activation, name = 'true_output')(conv9)
    
    model = Model(inputs = inputs, outputs = conv10)
    
    return model


# Use the following model for segmentation:
def unet_seg(input_size = (256,32,1)):
    '''
    Cell segmentation U-Net definition function.

    Parameters
    ----------
    input_size : tuple of 3 ints, optional
        Dimensions of the input tensor, without batch size.
        The default is (256,32,1).

    Returns
    -------
    model : Model
        Segmentation U-Net (compiled).

    '''
    
    model = unet(input_size = input_size, final_activation = 'sigmoid', output_classes = 1)
    model.compile(optimizer = Adam(lr = 1e-4), loss = pixelwise_weighted_binary_crossentropy, metrics = [unstack_acc])

    return model


# Use the following model for tracking and lineage reconstruction:
def unet_track(input_size = (256,32,4), class_weights = (1., 1., 1.)):
    '''
    Tracking U-Net definition function.

    Parameters
    ----------
    input_size : tuple of 3 ints, optional
        Dimensions of the input tensor, without batch size.
        The default is (256,32,4).
    class_weights : tuple/list of floats, optional
        Relative weights of each class. If daughters are very uncommon or each
        single cell occupies a small part of the image, this can make the U-Net
        converge to non-trivial solutions. (A trivial solution would be to 
        systematically return a mask of 0s)
        The default is (1., 1., 1.).

    Returns
    -------
    model : Model
        Tracking U-Net (compiled).

    '''
    
    model = unet(input_size = input_size, final_activation = 'softmax', output_classes = 3)
    model.compile(optimizer=Adam(lr = 1e-4), loss=class_weighted_categorical_crossentropy(class_weights), metrics = ['categorical_accuracy'])

    return model


# Use the following model for segmentation:
def unet_chambers(input_size = (512,512,1)):
    '''
    Chambers segmentation U-Net.

    Parameters
    ----------
    input_size : tuple of 3 ints, optional
        Dimensions of the input tensor, without batch size.
        The default is (512,512,1).

    Returns
    -------
    model : Model
        Chambers ID U-Net (compiled).

    '''
    
    model = unet(input_size = input_size, final_activation = 'sigmoid', output_classes = 1)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy')

    return model