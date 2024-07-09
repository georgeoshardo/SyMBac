import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops import array_ops, math_ops


def pixelwise_weighted_binary_crossentropy(y_true, y_pred):
    """
    Adapted from DeLTA paper
    https://gitlab.com/delta-microscopy/delta
    """

    seg = y_true[..., :-1]  # Assuming the mask is all but the last channel
    weight = y_true[..., -1:]  # Assuming the weight map is the last channel

    epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
    cond = y_pred >= zeros
    relu_logits = math_ops.select(cond, y_pred, zeros)
    neg_abs_logits = math_ops.select(cond, -y_pred, y_pred)
    entropy = math_ops.add(
        relu_logits - y_pred * seg, math_ops.log1p(math_ops.exp(neg_abs_logits)), name=None
    )

    # This is essentially the only part that is different from the Keras code:
    return K.mean(math_ops.multiply(weight, entropy), axis=-1)
