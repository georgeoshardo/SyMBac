import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.ndimage import distance_transform_edt, rotate, shift
from skimage.measure import label
from tensorflow.keras import backend as K


def tf_rotate(image, angle, order=1):  # Order = 1 for image, 0 for mask

    def rotate_image(image, angle, order):
        # Convert tensors to numpy arrays
        image_np = image.numpy()

        image_rotated = rotate(
            image_np, angle, axes=(1, 2), reshape=False, mode="nearest", order=order
        )

        return image_rotated

    image_rotated = tf.py_function(rotate_image, [image, angle, order], tf.float32)

    # Ensure that the output tensor has the same shape as the input
    image_rotated.set_shape(image.get_shape())

    return image_rotated


def tf_illumination_voodoo(image, num_control_points=5):
    """
    Adapted from DeLTA: https://gitlab.com/delta-microscopy/delta
    to be done using Tensorflow tensors
    """
    # Ensure image tensor is float32
    image = tf.cast(image, tf.float32)
    # Generate control points and corresponding random points
    control_points = tf.linspace(
        0.0, tf.cast(tf.shape(image)[1], tf.float32) - 1.0, num_control_points
    )
    random_points = tf.random.uniform([num_control_points], minval=0.1, maxval=0.9)
    # Generate a linear interpolation of the control points along the width of the image
    interp_points = tf.linspace(
        0.0, tf.cast(tf.shape(image)[1], tf.float32) - 1.0, tf.shape(image)[1]
    )
    curve = tfp.math.interp_regular_1d_grid(
        interp_points,
        control_points[0],
        control_points[-1],
        random_points,
        fill_value="constant_extension",
    )

    # Reshape and tile the curve to match the image shape for multiplication, adding an extra dimension to match the image shape
    curve = tf.reshape(curve, [1, -1, 1])
    curve = tf.tile(curve, [tf.shape(image)[0], 1, tf.shape(image)[2]])
    curve = tf.expand_dims(
        curve, -1
    )  # Add a dimension to curve to match the image's shape [?, 192, 32, 1]

    # Apply the curve to adjust image illumination
    newimage = image * curve

    return newimage


def tf_histogram_voodoo(image, num_control_points=3):
    """
    Adapted from DeLTA: https://gitlab.com/delta-microscopy/delta
    to be done using Tensorflow tensors
    Perform an approximation of histogram manipulation to simulate illumination changes,
    assuming the image is already normalized to [0, 1].
    """
    # Generate control points for the original and target histograms
    original_points = tf.linspace(0.0, 1.0, num=num_control_points + 2)
    target_points = tf.linspace(0.0, 1.0, num=num_control_points + 2)

    # Modify the target points with random values between 0.1 and 0.9 for the inner points
    random_values = tf.random.uniform([num_control_points], 0.1, 0.9)
    sorted_random_values = tf.sort(random_values)  # Ensure monotonicity
    target_points = tf.tensor_scatter_nd_update(
        target_points,
        tf.reshape(tf.range(1, num_control_points + 1), [-1, 1]),
        sorted_random_values,
    )

    # Assuming image has shape [height, width, channels], flatten it for interpolation, then reshape
    flat_image = tf.reshape(image, [-1])
    # Linearly interpolate the new image values based on the original and target histograms
    interp_values = tfp.math.interp_regular_1d_grid(
        flat_image,
        original_points[0],
        original_points[-1],
        target_points,
        fill_value="constant_extension",
    )

    # Reshape interpolated values back to the original image shape
    new_image = tf.reshape(interp_values, tf.shape(image))

    return new_image


def tf_add_gaussian_noise(image, gaussian_noise_max=0.1):

    # Uniformly sample the standard deviation of the Gaussian noise
    noise_stddev = tf.random.uniform((), 0, gaussian_noise_max)

    # Add Gaussian noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_stddev)
    noisy_image = image + noise

    # Clip the noisy image to ensure values remain in [0, 1]
    noisy_image_clipped = tf.clip_by_value(noisy_image, 0, 1)

    return noisy_image_clipped


def tf_apply_gaussian_blur(image, gaussian_blur_sigma_max=3.0):
    """Applies Gaussian blur to an image tensor."""

    def gaussian_kernel(size: int, mean: float, std: float, channels: int):
        """Generates a 2D Gaussian kernel for each channel."""
        d = tfp.distributions.Normal(mean, std)
        vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
        gauss_kernel = tf.einsum("i,j->ij", vals, vals)
        gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
        # Expand dimensions to [height, width, in_channels, channel_multiplier]
        gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
        return tf.tile(gauss_kernel, [1, 1, channels, 1])  # Tile for each channel

    # Ensure the image is a 4D tensor [batch_size, height, width, channels]
    if len(image.shape) == 3:
        image = image[tf.newaxis, ...]

    channels = image.shape[-1]
    # Sample sigma for Gaussian blur
    sigma = tf.random.uniform((), 0, gaussian_blur_sigma_max)
    # Determine kernel size based on sigma
    size = tf.cast(tf.math.ceil(sigma * 3), tf.int32)
    # Generate Gaussian kernel
    gauss_kernel = gaussian_kernel(size, 0.0, sigma, channels)

    # Apply Gaussian blur using depthwise_conv2d
    blurred_image = tf.nn.depthwise_conv2d(
        image, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME"
    )

    # Remove the batch dimension if it was added to a single image
    if blurred_image.shape[0] == 1:
        blurred_image = blurred_image[0]

    return blurred_image


def tf_shift(image, shift_fraction_y, shift_fraction_x, order=1):

    def shift_image(image, shift_fraction_y, shift_fraction_x, order):
        # Convert tensors to numpy arrays
        image_np = image.numpy()

        image_shift_px = np.array(
            [image_np.shape[1] * shift_fraction_y, image_np.shape[2] * shift_fraction_x],
            dtype=np.float32,
        )

        # Apply shifts
        image_shifted = shift(
            image_np, [0, image_shift_px[0], image_shift_px[1], 0], order=order, mode="nearest"
        )

        return image_shifted

    image_shifted = tf.py_function(
        shift_image, [image, shift_fraction_y, shift_fraction_x, order], tf.float32
    )

    # Ensure that the output tensor has the same shape as the input
    image_shifted.set_shape(image.get_shape())

    return image_shifted


def compute_weight_map_single(y, w0=10, sigma=5, wc=((0, 1), (1, 5))):
    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.



    Parameters
    ----------
    mask: Numpy array
        2D array of shape (image_height, image_width) representing binary mask
        of objects.
    wc: dict
        Dictionary of weight classes.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.

    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).



    References
    ----------
    Taken from the original U-net paper [1]_

    .. [1] Ronneberger, O., Fischer, P., Brox, T. (2015).
       U-Net: Convolutional Networks for Biomedical Image Segmentation.
       In: Navab, N., Hornegger, J., Wells, W., Frangi, A. (eds)
       Medical Image Computing and Computer-Assisted Intervention –
       MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science(),
       vol 9351. Springer, Cham. https://doi.org/10.1007/978-3-319-24574-4_28

    """
    # wc = {
    #    0: 1, # background
    #    1: 5  # objects
    # }

    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:, :, i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:, :, 0]
        d2 = distances[:, :, 1]
        w = w0 * np.exp(-1 / 2 * ((d1 + d2) / sigma) ** 2) * no_labels
    else:
        w = np.zeros_like(y)
    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc:  # wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w.astype(np.float32)


def tf_unet_weight_map(y, w0, sigma):
    # Assumes y has shape [batch_size, height, width, channels] and channels=1
    # Extract the first channel if it's grayscale image
    y_channel = tf.squeeze(y, axis=-1)  # Now y_channel has shape [batch_size, height, width]

    # Use tf.map_fn to apply the function to each item in the batch
    batch_weight_maps = tf.map_fn(
        lambda x: tf.numpy_function(compute_weight_map_single, [x, w0, sigma], Tout=tf.float32),
        y_channel,
        fn_output_signature=tf.float32,
    )

    batch_weight_maps = tf.expand_dims(batch_weight_maps, -1)
    return batch_weight_maps


def scaling_input(input_image, output_size):
    x = tf.image.resize(input_image, output_size, method="area")
    max_value = tf.reduce_max(x)
    min_value = tf.reduce_min(x)
    x = x - min_value
    return x / max_value


def scaling_mask(input_image, output_size):
    x = tf.image.resize(input_image, output_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    max_value = tf.reduce_max(x)
    return x / max_value

    return scaled_y


def scaling_output(input_image, output_size):
    x = tf.image.resize(input_image, output_size, method="area")
    max_value = tf.reduce_max(x)
    min_value = tf.reduce_min(x)
    x = x - min_value
    return x / max_value

    return scaled_y
