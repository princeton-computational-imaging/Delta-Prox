import os
import random

import numpy as np
import torch
from dprox.algo.admm import ADMM
from dprox.algo.specialization.rl.solver import (Batch, Env, apply_recursive,
                                          complex2channel)
from dprox.utils import to_torch_tensor, fft2, ifft2, hf
from PIL import Image
from scipy.io import loadmat
from tfpnp.data.util import data_augment, scale_height, scale_width
from torch.utils.data.dataset import Dataset
from torchlight.data import SingleImageDataset


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


def load_data(path):
    mat = loadmat(path)
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

    mat['params'] = {'y': mat['y0'], 'mask': mat['mask']}
    return mat


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


class TrainDataset(CSMRIDataset):
    def __getitem__(self, index):
        dic = super().__getitem__(index)
        dic['params'] = {'y': dic['y0'], 'mask': dic['mask']}
        return dic


class EvalDataset(CSMRIEvalDataset):
    def __getitem__(self, index):
        dic = super().__getitem__(index)
        dic['params'] = {'y': dic['y0'], 'mask': dic['mask']}
        return dic


class CustomADMM(ADMM):
    def _iter(self, state, rho, lam):
        x, z, u = state
        x = [x]
        z = z[0]

        for i, fn in enumerate(self.psi_fns):
            x[i] = fn.prox(z - u[i], lam=lam[fn])

        b = [x[i] + u[i] for i in range(len(self.psi_fns))]
        z = self.least_square.solve(b, rho)

        for i, fn in enumerate(self.psi_fns):
            u[i] = u[i] + x[i] - z

        return x[0], [z], u


class CustomEnv(Env):
    def __init__(self, data_loader, solver, max_episode_step):
        super().__init__(data_loader, solver, max_episode_step)
        self.channels = 0

    def get_policy_ob(self, ob):
        chs = self.channels
        vars = torch.split(ob.variables,
                           ob.variables.shape[1] // self.solver.state_dim, dim=1)
        vars = [v[:, chs:chs + 1, :, :] for v in vars]
        variables = torch.cat(vars, dim=1)
        x0 = ob.x0[:, chs:chs + 1, :, :]
        ob = torch.cat([
            variables,
            complex2channel(ob.y0),
            x0,
            ob.mask,
            ob.T,
            ob.sigma_n,
        ], 1).real
        return ob

    def _build_next_ob(self, ob, solver_state):
        return Batch(gt=ob.gt,
                     x0=ob.x0,
                     y0=ob.y0,
                     params=ob.params,
                     variables=solver_state,
                     mask=ob.mask,
                     sigma_n=ob.sigma_n,
                     T=ob.T + 1 / self.max_episode_step)

    def _observation(self):
        idx_left = self.idx_left
        params = apply_recursive(lambda x: x[idx_left, ...], self.state['params'])
        ob = Batch(gt=self.state['gt'][idx_left, ...],
                   x0=self.state['x0'][idx_left, ...],
                   y0=self.state['y0'][idx_left, ...],
                   params=params,
                   variables=self.state['solver'][idx_left, ...],
                   mask=self.state['mask'][idx_left, ...].float(),
                   sigma_n=self.state['sigma_n'][idx_left, ...],
                   T=self.state['T'][idx_left, ...])
        return ob


def custom_policy_ob_pack_fn(variables, x0, T, aux_state):
    return torch.cat([variables,
                      complex2channel(aux_state['y0']).cuda(),
                      x0,
                      aux_state['mask'].cuda(),
                      T,
                      aux_state['sigma_n'].cuda(),
                      ], dim=1).real


def sample(name='Bust.jpg'):
    mask = loadmat(hf.load_path('data/csmri/masks/radial_128_2.mat')).get('mask')
    mask = mask.astype('bool')

    imgpath = hf.load_path(os.path.join('data/csmri/Medical_128', name))
    target = Image.open(imgpath).convert('L')
    target = np.array(target, dtype=np.float32) / 255.0

    target = target[None]
    target = torch.from_numpy(target)
    mask = torch.from_numpy(mask)

    y0 = fft2(target)
    y0[:, ~mask] = 0
    Aty0 = ifft2(y0)

    # TODO: all sample should return [1, C, H, W] tensor, or [H, W, C] numpy
    y0 = y0.squeeze()  # [1, H, W] -> [H, W]
    Aty0 = Aty0.squeeze()
    target = target.squeeze()

    Aty0 = to_torch_tensor(Aty0, batch=True)
    y0 = to_torch_tensor(y0, batch=True)
    target = to_torch_tensor(target, batch=True)
    mask = to_torch_tensor(mask.unsqueeze(0), batch=True)
    return Aty0.real, y0, target, mask


class Dataset(SingleImageDataset):
    def __init__(self, root):
        super().__init__(root, mode='gray')
        mask = loadmat(hf.load_path('data/csmri/masks/radial_128_2.mat')).get('mask')
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
