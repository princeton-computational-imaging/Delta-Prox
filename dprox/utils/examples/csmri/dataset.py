import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat

from tfpnp.data.util import scale_height, scale_width, data_augment

from .misc import ifft2, fft2


class CSMRIDataset(Dataset):
    def __init__(self, datadir, fns, masks, noise_model=None, size=None, target_size=None, repeat=1, augment=False):
        super().__init__()
        self.datadir = datadir
        self.fns = fns or [im for im in os.listdir(self.datadir) if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png") or im.endswith(".tif")]
        self.fns = sorted(self.fns)
        self.masks = masks
        self.noise_model = noise_model
        self.size = size
        self.repeat = repeat
        self.target_size = target_size
        self.augment = augment

    def __getitem__(self, index):
        mask = self.masks[np.random.randint(0, len(self.masks))]
        mask = mask.astype(np.bool)

        sigma_n = 0

        index = index % len(self.fns)
        imgpath = os.path.join(self.datadir, self.fns[index])
        target = Image.open(imgpath).convert('L')

        if self.target_size is not None:
            ow, oh = target.size
            target = scale_height(target, self.target_size) if ow >= oh else scale_width(target, self.target_size)

        target = np.array(target, dtype=np.float32) / 255.0

        if target.ndim == 2:
            target = target[None]
        elif target.ndim == 3:
            target = target.transpose((2, 0, 1))
        else:
            raise NotImplementedError

        if self.augment:
            target = data_augment(target)

        target = torch.from_numpy(target)
        mask = torch.from_numpy(mask)

        y0 = fft2(target)
        y_clean = y0.clone()
        if self.noise_model is not None:
            y0, sigma_n = self.noise_model(y0)

        y0[:, ~mask] = 0

        ATy0 = ifft2(y0)
        x0 = ATy0.clone().detach()

        output = ATy0.real
        mask = mask.unsqueeze(0).bool()
        sigma_n = np.ones_like(y0) * sigma_n

        dic = {'y0': y0, 'x0': x0, 'ATy0': ATy0, 'gt': target, 'mask': mask, 'sigma_n': sigma_n.real, 'output': output, 'input': x0,
               'y_clean': y_clean}

        # y0,x0,ATy0, sigma_n: C, W, H, 2
        # gt, output: C, W, H
        # mask: 1, W, H

        return dic

    def __len__(self):
        if self.size is None:
            return len(self.fns) * self.repeat
        else:
            return self.size


def complex2real(x):
    return x[..., 0]


def view_as_complex(x):
    return x[..., 0] + 1j * x[..., 1]


class CSMRIEvalDataset(Dataset):
    def __init__(self, datadir, fns=None):
        super(CSMRIEvalDataset, self).__init__()
        self.datadir = datadir
        self.fns = fns or [im for im in os.listdir(self.datadir) if im.endswith(".mat")]
        self.fns.sort()

    def __getitem__(self, index):
        fn = self.fns[index]
        mat = loadmat(os.path.join(self.datadir, fn))

        mat['name'] = mat['name'].item()
        mat.pop('__globals__', None)
        mat.pop('__header__', None)
        mat.pop('__version__', None)
        mat['output'] = complex2real(mat['ATy0'])
        mat['input'] = view_as_complex(mat['x0'])
        mat['x0'] = view_as_complex(mat['x0'])
        mat['y0'] = view_as_complex(mat['y0'])
        mat['mask'] = np.expand_dims(mat['mask'], axis=0).astype('bool')
        mat['sigma_n'] = complex2real(mat['sigma_n'])

        return mat

    def __len__(self):
        return len(self.fns)


if __name__ == '__main__':
    from tfpnp.utils.noise import GaussianModelD
    from pathlib import Path
    data_dir = Path('data')
    mask_dir = Path('data/csmri/masks')
    train_root = data_dir / 'Images_128'

    sigma_ns = [5, 10, 15]
    sampling_masks = ['radial_128_2', 'radial_128_4', 'radial_128_8']

    noise_model = GaussianModelD(sigma_ns)
    masks = [loadmat(mask_dir / f'{sampling_mask}.mat').get('mask') for sampling_mask in sampling_masks]
    train_dataset = CSMRIDataset(train_root, fns=None, masks=masks, noise_model=noise_model)

    dic = train_dataset.__getitem__(0)
