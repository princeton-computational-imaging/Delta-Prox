import os
from pathlib import Path

import numpy as np
import torch
import torch.fft
from PIL import Image
from scipy.io import loadmat
from torchlight.data import SingleImageDataset

from dprox.utils import to_torch_tensor


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def fft2(x):
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, norm='ortho')
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x


def ifft2(x):
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, norm='ortho')
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x


def sample(name='Bust.jpg'):
    base_dir = Path(os.path.join(CURRENT_DIR, 'data'))

    mask = loadmat(base_dir / 'masks/radial_128_2.mat').get('mask')
    mask = mask.astype('bool')

    imgpath = base_dir / 'Medical_128' / name
    target = Image.open(imgpath).convert('L')
    target = np.array(target, dtype=np.float32) / 255.0

    target = target[None]
    target = torch.from_numpy(target)
    mask = torch.from_numpy(mask)

    y0 = fft2(target)
    y0[:, ~mask] = 0
    Aty0 = ifft2(y0)

    y0 = y0.squeeze()  # [1, H, W] -> [H, W]
    Aty0 = Aty0.squeeze()
    target = target.squeeze()

    Aty0 = to_torch_tensor(Aty0, batch=True)
    y0 = to_torch_tensor(y0, batch=True)
    target = to_torch_tensor(target, batch=True)
    mask = to_torch_tensor(mask, batch=True)
    return Aty0.real, y0, target, mask


class Dataset(SingleImageDataset):
    def __init__(self, root):
        super().__init__(root, mode='gray')
        base_dir = Path(os.path.join(CURRENT_DIR, 'data'))

        mask = loadmat(base_dir / 'masks/radial_128_2.mat').get('mask')
        mask = mask.astype('bool')

        self.mask = mask

    def __getitem__(self, index):
        target, path = super().__getitem__(index)
        target = torch.from_numpy(target)
        mask = torch.from_numpy(self.mask)

        y0 = fft2(target)
        y0[:, ~mask] = 0
        Aty0 = ifft2(y0)

        y0 = y0  # [1, H, W] -> [H, W]
        x0 = Aty0.real
        target = target

        return target, x0, y0