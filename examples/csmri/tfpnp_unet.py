import os

import torch
from scipy.io import loadmat
from tfpnp.utils.metric import psnr_qrnn3d
from torch.utils.data import DataLoader
from torchlight.logging import Logger

from dprox import *
from dprox.algo.tune import *
from dprox.utils import *
from dprox.utils.examples.csmri.common import (CustomADMM, CustomEnv,
                                               EvalDataset, TrainDataset,
                                               custom_policy_ob_pack_fn)


def get_datasets():
    from pathlib import Path

    from tfpnp.utils.noise import GaussianModelD
    data_dir = Path('data')
    mask_dir = Path('data/csmri/masks')
    train_root = data_dir / 'Images_128'

    sigma_ns = [5, 10, 15]
    sampling_masks = ['radial_128_2', 'radial_128_4', 'radial_128_8']

    noise_model = GaussianModelD(sigma_ns)
    masks = [loadmat(mask_dir / f'{sampling_mask}.mat').get('mask')
             for sampling_mask in sampling_masks]
    dataset = TrainDataset(train_root, fns=None, masks=masks, noise_model=noise_model)

    valid_datasets = {}
    for name in ['Medical7_2020', 'MICCAI_2020']:
        for ratio in [2, 4, 8]:
            for sigma in [5, 10, 15]:
                valid_datasets[f'{name}/radial_128_{ratio}/{sigma}'] = EvalDataset(f'data/csmri/{name}/radial_128_{ratio}/{sigma}')

    valid_datasets = {
        # 'Medical7_2020/radial_128_2/5': EvalDataset('data/csmri/Medical7_2020/radial_128_2/5'),
        # 'Medical7_2020/radial_128_2/10': EvalDataset('data/csmri/Medical7_2020/radial_128_2/10'),
        'Medical7_2020/radial_128_4/5': EvalDataset('data/csmri/Medical7_2020/radial_128_4/5'),
        'Medical7_2020/radial_128_4/15': EvalDataset('data/csmri/Medical7_2020/radial_128_4/15'),
        # 'Medical7_2020/radial_128_8/15': EvalDataset('data/csmri/Medical7_2020/radial_128_8/15'),
        'MICCAI_2020/radial_128_4/5': EvalDataset('data/csmri/MICCAI_2020/radial_128_4/5'),
        'MICCAI_2020/radial_128_8/5': EvalDataset('data/csmri/MICCAI_2020/radial_128_8/5'),
        # 'MICCAI_2020/radial_128_8/15': EvalDataset('data/csmri/MICCAI_2020/radial_128_8/15'),
    }
    return dataset, valid_datasets


def main():

    x = Variable()
    y = Placeholder()
    mask = Placeholder()
    data_term = csmri(x, mask, y)
    reg_term = deep_prior(x, denoiser='unet')
    solver = CustomADMM([reg_term], [data_term]).cuda()

    tf_solver = AutoTuneSolver(solver, policy='resnet', custom_policy_ob_pack_fn=custom_policy_ob_pack_fn)

    placeholders = {'y': y, 'mask': mask}

    # dataset
    dataset, valid_datasets = get_datasets()

    def step_fn(batch):
        y.value = batch['y0'].cuda()
        mask.value = batch['mask'].cuda()
        target = batch['gt'].cuda()
        x0 = batch['x0'].cuda()
        pred = tf_solver.solve(x0, batch).real
        return target, pred

    ckpt_path = 'ckpt/tfpnp_unet/actor_best.pkl'
    tf_solver.load(ckpt_path)
    tf_solver.eval(ckpt_path, valid_datasets, placeholders, custom_env=CustomEnv)

    save_root = 'abc/tfpnp_drunet'

    for name, valid_dataset in valid_datasets.items():
        total_psnr = 0
        test_loader = DataLoader(valid_dataset)

        save_dir = os.path.join(save_root, name)
        os.makedirs(save_dir, exist_ok=True)
        logger = Logger(save_dir, name=name)

        for idx, batch in enumerate(test_loader):
            with torch.no_grad():
                target, pred = step_fn(batch)
            psnr = psnr_qrnn3d(target.squeeze(0).cpu().numpy(),
                               pred.squeeze(0).cpu().numpy(),
                               data_range=1)
            total_psnr += psnr
            logger.save_img(f'{idx}.png', pred)

        logger.info('{} avg psnr= {}'.format(name, total_psnr / len(test_loader)))


main()
