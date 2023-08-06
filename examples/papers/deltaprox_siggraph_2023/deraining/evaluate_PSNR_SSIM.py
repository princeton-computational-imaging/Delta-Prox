import os

import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.color import rgb2ycbcr


def compute_ssim(img1, img2):
    if img1.shape[2] == 3:
        img1 = rgb2ycbcr(img1)[:, :, 0]

    if img2.shape[2] == 3:
        img2 = rgb2ycbcr(img2)[:, :, 0]

    ssim_mean = SSIM_index(img1, img2)
    return ssim_mean


def compute_psnr(img1, img2):
    if img1.shape[2] == 3:
        img1 = rgb2ycbcr(img1)[:, :, 0]

    if img2.shape[2] == 3:
        img2 = rgb2ycbcr(img2)[:, :, 0]

    imdff = img1.astype(np.float64) - img2.astype(np.float64)
    imdff = imdff.flatten()
    rmse = np.sqrt(np.mean(np.square(imdff)))
    psnr = 20 * np.log10(255 / rmse)
    return psnr


def SSIM_index(img1, img2, K=None, window=None, L=None):
    if img1.shape != img2.shape:
        return -np.inf, -np.inf

    M, N = img1.shape

    if K is None or len(K) == 0:
        K = [0.01, 0.03]
    if window is None or window.size == 0:
        if M < 11 or N < 11:
            return -np.inf, -np.inf
        window = np.outer(np.hanning(11), np.hanning(11))
        window /= np.sum(window)
    if L is None or L == 0:
        L = 255

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    mu1 = convolve(img1, window, mode='constant')
    mu2 = convolve(img2, window, mode='constant')
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = convolve(img1 ** 2, window, mode='constant') - mu1_sq
    sigma2_sq = convolve(img2 ** 2, window, mode='constant') - mu2_sq
    sigma12 = convolve(img1 * img2, window, mode='constant') - mu1_mu2

    if C1 > 0 and C2 > 0:
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        ssim_map = np.ones_like(mu1)
        index = (denominator1 * denominator2 > 0)
        ssim_map[index] = (numerator1[index] * numerator2[index]) / (denominator1[index] * denominator2[index])
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

    mssim = np.mean(ssim_map)

    return mssim


def evaluate(
    input_root='results/dprox_pdg_unroll/', 
    gt_root='datasets/test/', 
    datasets = ['Rain100H']
):
    num_set = len(datasets)

    psnr_alldatasets = 0
    ssim_alldatasets = 0

    for idx_set in range(num_set):
        file_path = os.path.join(input_root, datasets[idx_set])
        gt_path = os.path.join(gt_root, datasets[idx_set], 'target')
        path_list = [file for file in os.listdir(file_path) if file.endswith('.jpg') or file.endswith('.png')]
        gt_list = [file for file in os.listdir(gt_path) if file.endswith('.jpg') or file.endswith('.png')]
        img_num = len(path_list)

        total_psnr = 0
        total_ssim = 0

        if img_num > 0:
            for j in range(img_num):
                image_name = path_list[j]
                gt_name = gt_list[j]
                input = cv2.imread(os.path.join(file_path, image_name))
                gt = cv2.imread(os.path.join(gt_path, gt_name))
                ssim_val = compute_ssim(input, gt)
                psnr_val = compute_psnr(input, gt)
                total_ssim += ssim_val
                total_psnr += psnr_val

        qm_psnr = total_psnr / img_num if img_num > 0 else 0
        qm_ssim = total_ssim / img_num if img_num > 0 else 0

        print(f"For {datasets[idx_set]} dataset PSNR: {qm_psnr} SSIM: {qm_ssim}")

        psnr_alldatasets += qm_psnr
        ssim_alldatasets += qm_ssim


if __name__ == '__main__':
    import fire
    fire.Fire(evaluate)
    
    