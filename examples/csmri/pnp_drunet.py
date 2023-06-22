import os

import torch
from tfpnp.utils.metric import psnr_qrnn3d
from torch.utils.data import DataLoader
from torchlight.logging import Logger

from dprox import *
from dprox.algo.tune import *
from dprox.utils import *
from dprox.utils.examples.csmri.common import CustomADMM, EvalDataset


def main():

    x = Variable()
    y = Placeholder()
    mask = Placeholder()
    data_term = csmri(x, mask, y)
    reg_term = deep_prior(x, denoiser='drunet')
    solver = CustomADMM([reg_term], [data_term]).cuda()

    def step_fn(batch):
        y.value = batch['y0'].cuda()
        mask.value = batch['mask'].cuda()
        target = batch['gt'].cuda()
        x0 = batch['x0'].cuda()
        max_iter = 24

        # Medical7_2020/radial_128_2/5
        # rhos, sigmas = log_descent(50, 40, max_iter)
        # rhos, _ = log_descent(125, 0.1, max_iter)

        # Medical7_2020/radial_128_2/10
        # rhos, sigmas = log_descent(40, 40, max_iter)
        # rhos, _ = log_descent(0.1, 0.1, max_iter)

        # Medical7_2020/radial_128_4/5
        rhos, sigmas = log_descent(80, 40, max_iter)
        rhos, _ = log_descent(10, 0.1, max_iter)

        # Medical7_2020/radial_128_4/15
        # rhos, sigmas = log_descent(80, 55, max_iter)
        # rhos, _ = log_descent(1, 0.1, max_iter)

        # MICCAI_2020/radial_128_4/5
        # rhos, sigmas = log_descent(60, 40, max_iter)
        # rhos, _ = log_descent(4, 0.1, max_iter)

        # MICCAI_2020/radial_128_8/5
        # rhos, sigmas = log_descent(60, 40, max_iter)
        # rhos, _ = log_descent(1, 0.1, max_iter)

        # rhos, sigmas = log_descent(50, 1, max_iter)
        pred = solver.solve(x0=x0, rhos=rhos, lams={reg_term: sigmas}, max_iter=max_iter).real
        return target, pred

    valid_datasets = {
        # 'Medical7_2020/radial_128_2/5': EvalDataset('data/csmri/Medical7_2020/radial_128_2/5'),
        'Medical7_2020/radial_128_4/5': EvalDataset('data/csmri/Medical7_2020/radial_128_4/5'),
        # 'Medical7_2020/radial_128_2/10': EvalDataset('data/csmri/Medical7_2020/radial_128_2/10'),
        # 'Medical7_2020/radial_128_8/15': EvalDataset('data/csmri/Medical7_2020/radial_128_8/15'),
        # 'Medical7_2020/radial_128_4/15': EvalDataset('data/csmri/Medical7_2020/radial_128_4/15'),
        # 'MICCAI_2020/radial_128_4/5': EvalDataset('data/csmri/MICCAI_2020/radial_128_4/5'),
        # 'MICCAI_2020/radial_128_8/15': EvalDataset('data/csmri/MICCAI_2020/radial_128_8/15'),
    }

    save_root = 'abc/pnp_drunet'

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
