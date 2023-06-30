from torchlight.metrics import psnr
import imageio
import os

dir_a = 'results/DGUNet_restormer2_plus_tune_epoch32-2/Rain100H'
dir_b = 'results/DGUNet_conv/Rain100H'
dir_gt = 'datasets/test/Rain100H/target'
for name in os.listdir(dir_a):
    path_a = os.path.join(dir_a, name)
    path_b = os.path.join(dir_b, name)
    path_gt = os.path.join(dir_gt, name)
    
    image_a = imageio.imread(path_a)
    image_b = imageio.imread(path_b)
    image_gt = imageio.imread(path_gt)
    
    psnr1 = psnr(image_a, image_gt, data_range=255)
    psnr2 = psnr(image_b, image_gt, data_range=255)
    
    print(name, psnr1, psnr2)