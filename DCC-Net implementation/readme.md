The aim of this project is to employ the knowledge of computer vision and deep learning models capable of effectively denoising and enhancing images captured in extreme low light conditions. In this project i have developed and implemented convolutional neural networks DCENet and got ashtoninguishly competitive results.

after cloning or downloading these files:
for testing you have to create a folder named 'test' and two subfolder into this folder named 'low' and 'high' which stores the low light and the ground truth images of the test dataset respectively.

for training you have to create a folder named 'train' and two subfolder into this folder named 'low' and 'high' which stores the low light and the ground truth images of the train dataset respectively.

![directory](https://github.com/pukhraj102/low-light-image-denoising/assets/127439548/5e1a44cb-f7ab-4981-b58a-d43accde4852)


JUST RUN THE GIVEN LINE!

For performing training operation  of model

![train](https://github.com/pukhraj102/low-light-image-denoising/assets/127439548/fffd3fe9-cb97-4f3b-b25b-3e309aa5c92d)


For performing testing operation
![test](https://github.com/pukhraj102/low-light-image-denoising/assets/127439548/b313fe61-709e-4915-9f98-c2100e1b1933)


This will train the model if --task 0 is specified, or test the model if --task 1 is specified, with options to plot and save the results.
Along with this —save_dir can be used to specify where to save the denoised images, by default it is set to './test/predicted/' as asked in deliverables.
