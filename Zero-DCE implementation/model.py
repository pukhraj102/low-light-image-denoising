from keras import layers
#from skimage.restoration import denoise_wavelet
from losses import *


def dce_net():
    input_img = keras.Input(shape=[None, None, 3])
    conv1 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(input_img)
    # conv1 = layers.BatchNormalization()(conv1)
    # conv1 = layers.Dropout(0.2)(conv1)

    conv2 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv1)
    # conv2 = layers.BatchNormalization()(conv2)
    # conv2 = layers.Dropout(0.2)(conv2)

    conv3 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv2)
    # conv3 = layers.BatchNormalization()(conv3)
    # conv3 = layers.Dropout(0.2)(conv3)

    conv4 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv3)
    # conv4 = layers.BatchNormalization()(conv4)
    # conv4 = layers.Dropout(0.2)(conv4)

    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])

    conv5 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(int_con1)
    # conv5 = layers.BatchNormalization()(conv5)
    # conv5 = layers.Dropout(0.2)(conv5)

    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])

    conv6 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(int_con2)
    # conv6 = layers.BatchNormalization()(conv6)
    # conv6 = layers.Dropout(0.2)(conv6)

    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
    x_r = layers.Conv2D(24, (3, 3), strides=(1, 1), activation="tanh", padding="same")(int_con3)

    return keras.Model(inputs=input_img, outputs=x_r)


class ZeroDCE(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the DCE-Net model
        self.dce_model = dce_net()
        self.psnr = psnr
        self.mse = mse
        self.mae = mae

    def compile(self, learning_rate, **kwargs):
        super().compile(**kwargs)
        # Set the optimizer with the given learning rate
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        # Initialize the loss functions and metrics
        self.spatial_constancy_loss = SpatialConsistencyLoss()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.psnr_loss_tracker = keras.metrics.Mean(name="psnr")
        self.mse_loss_tracker = keras.metrics.Mean(name="mse")
        self.mae_loss_tracker = keras.metrics.Mean(name="mae")

    @property
    def metrics(self):
        # Return the list of metrics being tracked
        return [
            self.total_loss_tracker,
            self.mse_loss_tracker,
            self.mae_loss_tracker,
            self.psnr_loss_tracker,
        ]

    def get_enhanced_image(self, data, net_output):
        # Split the output into 8 different components
        r1 = net_output[:, :, :, :3]
        r2 = net_output[:, :, :, 3:6]
        r3 = net_output[:, :, :, 6:9]
        r4 = net_output[:, :, :, 9:12]
        r5 = net_output[:, :, :, 12:15]
        r6 = net_output[:, :, :, 15:18]
        r7 = net_output[:, :, :, 18:21]
        r8 = net_output[:, :, :, 21:24]

        # Apply the enhancement steps
        x = data + r1 * (tf.square(data) - data)
        x = x + r2 * (tf.square(x) - x)
        x = x + r3 * (tf.square(x) - x)
        x = x + r4 * (tf.square(x) - x)
        x = x + r5 * (tf.square(x) - x)
        x = x + r6 * (tf.square(x) - x)
        x = x + r7 * (tf.square(x) - x)
        enhanced_image = x + r8 * (tf.square(x) - x)

        return enhanced_image

    def call(self, data):
        # Get the output from the DCE-Net model
        dce_net_output = self.dce_model(data)
        # Return the enhanced image
        return self.get_enhanced_image(data, dce_net_output)

    def compute_losses(self, data, output):
        enhanced_image = self.get_enhanced_image(data, output)
        loss_illumination = 200 * illumination_smoothness_loss(output)
        loss_spatial_constancy = tf.reduce_mean(
            self.spatial_constancy_loss(enhanced_image, data)
        )
        loss_color_constancy = 5 * tf.reduce_mean(color_constancy_loss(enhanced_image))
        loss_exposure = 10 * tf.reduce_mean(exposure_loss(enhanced_image))
        total_loss = (
                loss_illumination
                + loss_spatial_constancy
                + loss_color_constancy
                + loss_exposure
        )

        return {
            "total_loss": total_loss,
            "illumination_smoothness_loss": loss_illumination,
            "spatial_constancy_loss": loss_spatial_constancy,
            "color_constancy_loss": loss_color_constancy,
            "exposure_loss": loss_exposure,
        }


    def train_step(self, dataset):
        data, true = dataset
        with tf.GradientTape() as tape:
            output = self.dce_model(data)
            losses = self.compute_losses(data, output)

        gradients = tape.gradient(
            losses["total_loss"], self.dce_model.trainable_weights
        )
        self.optimizer.apply_gradients(zip(gradients, self.dce_model.trainable_weights))

        self.total_loss_tracker.update_state(losses["total_loss"])

        mse_value = self.mse(true, self.get_enhanced_image(data, output))
        mae_value = self.mae(true, self.get_enhanced_image(data, output))
        psnr_value = self.psnr(true, self.get_enhanced_image(data, output))
        self.mse_loss_tracker.update_state(mse_value)
        self.mae_loss_tracker.update_state(mae_value)
        self.psnr_loss_tracker.update_state(psnr_value)

        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, dataset):
        data, true = dataset

        output = self.dce_model(data)
        losses = self.compute_losses(data, output)

        self.total_loss_tracker.update_state(losses["total_loss"])
        mse_value = self.mse(true, self.get_enhanced_image(data, output))
        mae_value = self.mae(true, self.get_enhanced_image(data, output))
        psnr_value = self.psnr(true, self.get_enhanced_image(data, output))

        self.mse_loss_tracker.update_state(mse_value)
        self.mae_loss_tracker.update_state(mae_value)
        self.psnr_loss_tracker.update_state(psnr_value)

        return {metric.name: metric.result() for metric in self.metrics}

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        """Save the weights of the DCE-Net model"""
        self.dce_model.save_weights(
            filepath,
            overwrite=overwrite,
        )

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        """Load the weights of the DCE-Net model"""
        self.dce_model.load_weights(
            filepath=filepath,
            skip_mismatch=skip_mismatch,
        )



# from keras import layers, models
# from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
# from keras.models import Model


# def DnCNN(depth = 10, num_filters=32):
#     # Encoder
#     input_img = layers.Input(shape=[None, None, 3])
#     #input_img = Input(shape=input_shape)

#     x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_img)
#     x = layers.Activation('relu')(x)

#     for _ in range(depth - 2):
#         x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)

#     x = layers.Conv2D(3, (3, 3), padding='same')(x)
#     output_img = Add()([input_img, x])  # Residual learning

#     model = Model(input_img, output_img)
#     return model