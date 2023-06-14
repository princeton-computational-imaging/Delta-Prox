import torch
import torch.nn as nn
from scipy.io import loadmat

from dprox import *
from dprox.algo.tune import *
from dprox.utils import *
from dprox.algo.special.deq.solver import DEQ

import argparse

from common import TrainDataset, EvalDataset, CustomADMM, CustomEnv, custom_policy_ob_pack_fn



class CustomDEQSolver(nn.Module):
    def __init__(self, solver: Algorithm, tf_solver):
        super().__init__()
        self.internal = solver

        def fn(z, x, aux_state):
            state = solver.unpack(z)
            rhos, lams, idx_stop = tf_solver.estimate(state, 0, state[0], aux_state)
            state = solver.iters(state, rhos, lams, max_iter=1)
            return solver.pack(state)
        self.solver = DEQ(fn)

    def forward(self, x0, aux_state, **kwargs):
        state = self.internal.initialize(x0)
        state = self.internal.pack(state)
        state = self.solver(state, aux_state)
        state = self.internal.unpack(state)
        return state[0]

    def solve(self, x0, rhos, lams, **kwargs):
        return self.forward(x0, rhos, lams, **kwargs)


def main():
    seed_everything(1234)

    x = Variable()
    y = Placeholder()
    mask = Placeholder()
    data_term = csmri(x, mask, y)
    reg_term = deep_prior(x, denoiser='unet', trainable=True, sqrt=False)
    # solver = compile(data_term + reg_term, method='ladmm')
    solver = CustomADMM([reg_term], [data_term]).cuda()

    placeholders = {'y': y, 'mask': mask}

    # dataset

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
        'Medical7_2020/radial_128_2/15': EvalDataset('data/csmri/Medical7_2020/radial_128_2/10'),
        # 'Medical7_2020/radial_128_4/15': EvalDataset('data/csmri/Medical7_2020/radial_128_4/15'),
        # 'Medical7_2020/radial_128_8/15': EvalDataset('data/csmri/Medical7_2020/radial_128_8/15'),
    }

    # train
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/Images_128')
    parser.add_argument('--savedir', type=str, default='deq_csmri_admm_iter1')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('-r', '--resume', type=str, default=None)
    args = parser.parse_args()

    tf_solver = AutoTuneSolver(solver, policy='resnet', action_pack=1, max_episode_step=30,
                               custom_policy_ob_pack_fn=custom_policy_ob_pack_fn)
    ckpt_path = 'ckpt/custom_admm_pack1-train2/actor_best.pkl'
    tf_solver.load(ckpt_path)
    tf_solver.eval(ckpt_path, valid_datasets, placeholders, custom_env=CustomEnv)
    deq_solver = CustomDEQSolver(solver, tf_solver)

    def step_fn(batch):
        y.value = batch['y0'].cuda()
        mask.value = batch['mask'].cuda()
        target = batch['gt'].cuda()
        x0 = batch['x0'].cuda()
        pred = deq_solver.forward(x0, batch).real
        # pred = tf_solver.solve(x0, batch).real
        return target, pred

    # train_deq(args, dataset, deq_solver, step_fn)

    ckpt = torch.load('ckpt/deq_csmri_admm_iter1/epoch_9.pth')
    deq_solver.load_state_dict(ckpt['solver'], strict=False)
    deq_solver.eval()
    from tfpnp.utils.metric import psnr_qrnn3d

    total_psnr = 0
    # test_loader = DataLoader(EvalDataset('data/csmri/Medical7_2020/radial_128_4/15'))
    test_loader = DataLoader(EvalDataset('data/csmri/MICCAI_2020/radial_128_4/5'))
    for batch in test_loader:
        with torch.no_grad():
            target, pred = step_fn(batch)
        psnr = psnr_qrnn3d(target.squeeze(0).cpu().numpy(),
                           pred.squeeze(0).cpu().numpy(),
                           data_range=1)
        total_psnr += psnr
    print('avg psnr=', total_psnr / len(test_loader))


main()
