import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
from torch.nn import DataParallel
from test import *
from train import *


parser = argparse.ArgumentParser(prog='Low light Image VLG by Pukhraj Choudhary')

parser.add_argument('--task', dest='task', type=int, default=1,
                    help='what do you want to do, 1 for test, 0 for training')
parser.add_argument('--save_enable', dest='save_enable', type=int, default=1,
                    help='toggle whether to save test image result, 1 for enable and 0 for disable')
parser.add_argument('--save_dir', dest='save_dir', default='./test/predicted/',
                    help='directory where you wants to save the resulting image.')
parser.add_argument('--plot_test', dest='plot_test', type=int, default=1,
                    help='1 for ploting test prediction and ground_truth, else 0')

args = parser.parse_args(['--task', '1' , '--plot_test', '1', '--save_enable', '1'])


def main():
    if args.task:
        predicted_path = args.save_dir
        os.makedirs(predicted_path, exist_ok=True)

        perform_test(predicted_path, args.save_enable)
        check_psnr(predicted_path,args.plot_test)


    else:
        # Define model and optimizer
        model = color_net()
        model = DataParallel(model)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)

        # Early stopping parameters
        patience = 5
        best_loss = float('inf')
        counter = 0

        # Train the model
        for epoch in range(100):
            running_loss = 0.0
            total_psnr = 0.0
            for i, (img_low, img_high) in enumerate(train_loader):
                # Forward pass
                img_low = img_low.view(batch_size, 3, 256, 256)
                gray, color_hist, img_enhance = model(img_low)
                img_enhance = img_enhance.view_as(img_high)
                loss = criterion(img_enhance, img_high)
                psnr = psnr_calc(img_high, img_enhance)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_psnr += psnr

            # Print statistics
            avg_loss = running_loss / len(train_loader)
            avg_psnr = total_psnr / len(train_loader)
            print(f'Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}, Avg PSNR: {avg_psnr:.2f}')

            # Validate the model
            with torch.no_grad():
                val_loss = 0.0
                val_psnr = 0.0
                for i, (img_low, img_high) in enumerate(val_loader):
                    img_low = img_low.view(val_batch_size, 3, 256, 256)
                    gray, color_hist, img_enhance = model(img_low)
                    img_enhance = img_enhance.view_as(img_high)
                    loss = criterion(img_enhance, img_high)
                    psnr = psnr_calc(img_high, img_enhance)

                    val_loss += loss.item()
                    val_psnr += psnr

                # Print statistics
                avg_val_loss = val_loss / len(val_loader)
                avg_val_psnr = val_psnr / len(val_loader)
                print(f'Epoch {epoch + 1}, Val Loss: {avg_val_loss:.4f}, Avg Val PSNR: {avg_val_psnr:.2f}')


            # Early stopping check
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

        torch.save(model.state_dict(), './weights.pth')


if __name__ == '__main__':
    main()
