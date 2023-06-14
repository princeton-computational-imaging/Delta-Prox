import argparse

import torch
from scipy.io import loadmat
from tfpnp.utils.metric import psnr_qrnn3d
from torchlight.logging import Logger

from dprox import *
from dprox.algo.tune import *
from dprox.utils import *


from common import CustomADMM, EvalDataset, TrainDataset


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
        'Medical7_2020/radial_128_2/5': EvalDataset('data/csmri/Medical7_2020/radial_128_2/5'),
        # 'Medical7_2020/radial_128_4/15': EvalDataset('data/csmri/Medical7_2020/radial_128_4/15'),
        # 'Medical7_2020/radial_128_8/15': EvalDataset('data/csmri/Medical7_2020/radial_128_8/15'),
    }
    return dataset, valid_datasets


def get_args(save_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/Images_128')
    parser.add_argument('--savedir', type=str, default=save_dir)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-r', '--resume', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    seed_everything(1234)

    x = Variable()
    y = Placeholder()
    mask = Placeholder()
    data_term = csmri(x, mask, y)
    reg_term = deep_prior(x, denoiser='unet', trainable=True)
    solver = CustomADMM([reg_term], [data_term]).cuda()
    deq_solver = DEQSolver(solver, learned_params=True).cuda()
    # deq_solver = DEQSolver(solver, learned_params=False).cuda()

    # save_dir = 'ckpt/deq_unet2_not_learn_param'
    save_dir = 'ckpt/deq_unet2'
    args = get_args(save_dir)
    dataset, valid_datasets = get_datasets()

    def step_fn(batch):
        y.value = batch['y0'].cuda()
        mask.value = batch['mask'].cuda()
        target = batch['gt'].cuda()
        x0 = batch['x0'].cuda()

        max_iter = 1
        rhos, _ = log_descent(1, 1, max_iter, lam=0.5)
        _, sigmas = log_descent(130, 130, max_iter)

        pred = deq_solver.solve(x0=x0, rhos=rhos, lams={reg_term: sigmas}, max_iter=max_iter).real
        # pred = solver.solve(x0=x0, rhos=rhos, lams={reg_term: sigmas}, max_iter=max_iter).real
        return target, pred

    if args.train:
        train_deq(args, dataset, deq_solver, step_fn)

    ckpt = torch.load(os.path.join(save_dir, 'last.pth'))
    deq_solver.load_state_dict(ckpt['solver'], strict=True)
    deq_solver.eval()

    valid_datasets = {
        # 'Medical7_2020/radial_128_2/5': EvalDataset('data/csmri/Medical7_2020/radial_128_2/5'),
        # 'Medical7_2020/radial_128_2/10': EvalDataset('data/csmri/Medical7_2020/radial_128_2/10'),
        'Medical7_2020/radial_128_4/5': EvalDataset('data/csmri/Medical7_2020/radial_128_4/5'),
        'Medical7_2020/radial_128_4/15': EvalDataset('data/csmri/Medical7_2020/radial_128_4/15'),

        'MICCAI_2020/radial_128_4/5': EvalDataset('data/csmri/MICCAI_2020/radial_128_4/5'),
        'MICCAI_2020/radial_128_8/5': EvalDataset('data/csmri/MICCAI_2020/radial_128_8/5'),
    }

    save_root = 'abc/deq_unet'

    for name, valid_dataset in valid_datasets.items():
        total_psnr = 0
        test_loader = DataLoader(valid_dataset)

        save_dir = os.path.join(save_root, name)
        os.makedirs(save_dir, exist_ok=True)
        logger = Logger(save_dir, name=name)

        for batch in test_loader:
            with torch.no_grad():
                target, pred = step_fn(batch)
            psnr = psnr_qrnn3d(target.squeeze(0).cpu().numpy(),
                               pred.squeeze(0).cpu().numpy(),
                               data_range=1)
            total_psnr += psnr
            logger.save_img(f'{psnr:0.2f}.png', pred)

        logger.info('{} avg psnr= {}'.format(name, total_psnr / len(test_loader)))


main()
