import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import tensorflow as tf


def color_constancy_loss(x):
    mean_rgb = tf.reduce_mean(x, axis=(1, 2), keepdims=True)

    # mean values for each color channel
    mr = mean_rgb[:, :, :, 0]
    mg = mean_rgb[:, :, :, 1]
    mb = mean_rgb[:, :, :, 2]

    d_rg = tf.square(mr - mg)
    d_rb = tf.square(mr - mb)
    d_gb = tf.square(mb - mg)

    loss = tf.sqrt(tf.square(d_rg) + tf.square(d_rb) + tf.square(d_gb))

    return loss


def exposure_loss(x, mean_val=0.5):
    x = tf.reduce_mean(x, axis=3, keepdims=True)
    mean = tf.nn.avg_pool2d(x, ksize=16, strides=16, padding="VALID")
    difference = tf.square(mean - mean_val)
    loss = tf.reduce_mean(difference)

    return loss


def illumination_smoothness_loss(x):
    batch_size = tf.shape(x)[0]
    height = tf.shape(x)[1]
    weight = tf.shape(x)[2]

    count_h = (tf.shape(x)[2] - 1) * tf.shape(x)[3]
    count_w = tf.shape(x)[2] * (tf.shape(x)[3] - 1)

    h_tv = tf.reduce_sum(tf.square((x[:, 1:, :, :] - x[:, : height - 1, :, :])))
    w_tv = tf.reduce_sum(tf.square((x[:, :, 1:, :] - x[:, :, : weight - 1, :])))

    batch_size = tf.cast(batch_size, dtype=tf.float32)
    count_h = tf.cast(count_h, dtype=tf.float32)
    count_w = tf.cast(count_w, dtype=tf.float32)

    loss = 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    return loss


class SpatialConsistencyLoss(keras.losses.Loss):
    def __init__(self):
        super().__init__(reduction="none")

        # Define the convolution kernels for spatial consistency checks
        self.left_kernel = tf.constant(
            [[[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.right_kernel = tf.constant(
            [[[[0, 0, 0]], [[0, 1, -1]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.up_kernel = tf.constant(
            [[[[0, -1, 0]], [[0, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.down_kernel = tf.constant(
            [[[[0, 0, 0]], [[0, 1, 0]], [[0, -1, 0]]]], dtype=tf.float32
        )

    def call(self, y_true, y_pred):
        # Calculate mean values across the color channels for both true and predicted images
        original_mean = tf.reduce_mean(y_true, axis=3, keepdims=True)
        enhanced_mean = tf.reduce_mean(y_pred, axis=3, keepdims=True)

        # Apply average pooling to downsample the images
        original_pool = tf.nn.avg_pool2d(original_mean, ksize=4, strides=4, padding="VALID")
        enhanced_pool = tf.nn.avg_pool2d(enhanced_mean, ksize=4, strides=4, padding="VALID")

        # Calculate spatial differences using convolution with the defined kernels
        d_original_left = tf.nn.conv2d(original_pool, self.left_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_original_right = tf.nn.conv2d(original_pool, self.right_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_original_up = tf.nn.conv2d(original_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_original_down = tf.nn.conv2d(original_pool, self.down_kernel, strides=[1, 1, 1, 1], padding="SAME")

        d_enhanced_left = tf.nn.conv2d(enhanced_pool, self.left_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_enhanced_right = tf.nn.conv2d(enhanced_pool, self.right_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_enhanced_up = tf.nn.conv2d(enhanced_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_enhanced_down = tf.nn.conv2d(enhanced_pool, self.down_kernel, strides=[1, 1, 1, 1], padding="SAME")

        # Compute the squared differences between original and enhanced spatial differences
        d_left = tf.square(d_original_left - d_enhanced_left)
        d_right = tf.square(d_original_right - d_enhanced_right)
        d_up = tf.square(d_original_up - d_enhanced_up)
        d_down = tf.square(d_original_down - d_enhanced_down)

        # Sum up the squared differences to get the final loss
        loss = d_left + d_right + d_up + d_down

        return loss


def psnr(true, pred):
    return tf.image.psnr(true, pred, max_val=1.0)


def mse(true, pred):
    return tf.reduce_mean(tf.square(true - pred))


def mae(true, pred):
    return tf.reduce_mean(tf.abs(true - pred))
