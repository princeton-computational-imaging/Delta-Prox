import argparse
import os

import imageio
import torch
import utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from derain.data import get_test_data
from derain.DGUNet_plus2 import DGUNet

parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')

parser.add_argument('--input_dir', default='./datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results2/DGUNet_restormer2_plus_tune_epoch32-2', type=str, help='Directory for results')
parser.add_argument('--weights', default='checkpoints/DGUNet_plus2/model_best.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()


model_restoration = DGUNet()

utils.load_checkpoint(model_restoration, args.weights)

model_restoration.cuda()
model_restoration.eval()

datasets = ['Rain100H']


for dataset in datasets:
    rgb_dir_test = os.path.join(args.input_dir, dataset, 'input')
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)

    result_dir = os.path.join(args.result_dir, dataset)
    utils.mkdir(result_dir)

    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            input_ = data_test[0].cuda()
            filenames = data_test[1]

            restored = model_restoration(input_)

            restored = torch.clamp(restored[0], 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy() * 255

            for batch in range(len(restored)):
                restored_img = restored[batch].astype('uint8')
                imageio.imsave(os.path.join(result_dir, filenames[batch] + '.png'),
                               restored_img)
