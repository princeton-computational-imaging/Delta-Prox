import argparse

import torch
from torchlight.data import SingleImageDataset

from dprox import *
from dprox.algo.tune import *
from dprox.utils import *
from dprox.utils import examples

img = sample('face')
psf = point_spread_function(15, 5)
b = blurring(img, psf)

x = Variable()
y = Placeholder()
data_term = sum_squares(conv(x, psf) - y)
reg_term = deep_prior(x, denoiser='ffdnet_color', trainable=True)
solver = compile(data_term + reg_term, method='admm')
solver = DEQSolver(solver)

y.value = b.cuda()
max_iter = 100
rhos, sigmas = log_descent(30, 30, max_iter)
b = b.cuda()
out = solver.solve(x0=b, rhos=rhos, lams={reg_term: sigmas}, max_iter=max_iter)

print(psnr(out, img))  
imshow(out)


# train
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='data/bsd500')
parser.add_argument('--savedir', type=str, default='saved_deconv')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--bs', type=int, default=8)
parser.add_argument('-r', '--resume', type=str, default=None)
args = parser.parse_args()


class Dataset(SingleImageDataset):
    def __init__(self, root):
        super().__init__(root, crop_size=(256, 256))

    def __getitem__(self, index):
        img, path = super().__getitem__(index)
        lr = scipy.ndimage.filters.convolve(img.transpose(1, 2, 0), psf, mode='wrap')
        lr = lr.transpose(2, 0, 1)
        return img, lr


dataset = Dataset(args.root)


def step_fn(batch):
    target, b = batch
    y.value = b.cuda()
    pred = solver.solve(x0=b.cuda(), rhos=rhos, lams={reg_term: sigmas}, max_iter=max_iter)
    return target.cuda(), pred


train_deq(args, dataset, solver, step_fn)
