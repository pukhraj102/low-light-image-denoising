import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import keras
from test import *
from train import *

parser = argparse.ArgumentParser(prog='Low light Image VLG by Pukhraj Choudhary')

parser.add_argument('--task', dest='task', type=int, default=1,
                    help='what do you want to do, 1 for test, 0 for training')
parser.add_argument('--save_enable', dest='save_enable', type=int, default=1,
                    help='toggle whether to save test image result, 1 for enable and 0 for disable')
parser.add_argument('--save_dir', dest='test_save', default='./test/predicted/',
                    help='directory where you wants to save the resulting image.')
parser.add_argument('--plot_test', dest='plot_test', type=int, default=1,
                    help='1 for ploting test prediction and ground_truth, else 0')

args = parser.parse_args(['--task', '1' , '--plot_test', '1', '--save_enable', '1'])


def main():
    if args.task:
        predicted_dir = args.test_save
        os.makedirs(predicted_dir, exist_ok=True)

        perform_test(predicted_dir, args.plot_test, args.save_enable)


    else:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_total_loss",
            min_delta=0.001,
            patience=15,
            verbose=0,
            mode="min",
            baseline=None,
            restore_best_weights=False,
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_psnr', factor=0.4, patience=8, min_lr=1e-6,
                                                         verbose=1)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=100000, decay_rate=0.96, staircase=True
        )

        zero_dce_model = ZeroDCE()
        zero_dce_model.compile(learning_rate=1e-2)
        history = zero_dce_model.fit(dataset, validation_data=val_dataset, epochs=100, verbose=1,
                                     callbacks=early_stopping)

        zero_dce_model.save_weights('./weights/model.weights.h5')

        # Plot val_total_loss and total_loss vs epoch
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['total_loss'], label='Total Loss')
        plt.plot(history.history['val_total_loss'], label='Validation Total Loss')
        plt.title('Total Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.legend()
        plt.show()

        # Plot val_psnr and psnr vs epoch
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['psnr'], label='PSNR')
        plt.plot(history.history['val_psnr'], label='Validation PSNR')
        plt.title('PSNR vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
