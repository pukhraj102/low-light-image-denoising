from utils import *
from glob import glob
from PIL import Image
test_low = sorted(glob("./test/low/*"))
test_high = sorted(glob("./test/high/*"))

psnr_values = []
mse_values = []
mae_values = []


def infer_and_save(original_image, output_path, zero_dce_model, save_enable):
    image = keras.utils.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = zero_dce_model(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())
    if save_enable:
        output_image.save(output_path)
    return output_image


def perform_test(predicted_dir, plot, save_enable):
    zero_dce_model = ZeroDCE()
    zero_dce_model.load_weights('./weights/model.weights.h5')

    for idx in range(len(test_low)):
        enhanced_image_output_path = os.path.join(predicted_dir, f"enhanced_{idx}.png")
        original_image = Image.open(test_low[idx])
        enhanced_image = infer_and_save(original_image, enhanced_image_output_path, zero_dce_model, save_enable)
        ground_truth = Image.open(test_high[idx])

        if plot:
            plot_results(
                [original_image, enhanced_image, ground_truth],
                ["Original", "Enhanced", "Ground Truth"],
                (12, 12),
            )

        PSNR = psnr_img(enhanced_image, ground_truth)
        MSE = mse_img(enhanced_image, ground_truth)
        MAE = mae_img(enhanced_image, ground_truth)
        psnr_values.append(PSNR)
        mse_values.append(MSE)
        mae_values.append(MAE)
        print("PSNR is: ", PSNR, "\tMSE IS: ", MSE, "\tMAE is: ", MAE, "\n")

    average_psnr = sum(psnr_values) / len(psnr_values)
    average_mse = sum(mse_values) / len(mse_values)
    average_mae = sum(mae_values) / len(mae_values)

    print(" Average PSNR is: ", average_psnr, "\nAverage MSE IS: ", average_mse, "\nAverage MAE is: ", average_mae,
          "\n")
