import os

import torch
import torch.nn as nn
from tfpnp.utils.metric import psnr_qrnn3d
from torch.utils.data import DataLoader

from dprox import *
from dprox.algo.special.deq.solver import DEQ
from dprox.algo.tune import *
from dprox.utils import *
from dprox.utils.examples.csmri.common import (CustomADMM, EvalDataset, custom_policy_ob_pack_fn)

# allowed torlence
TOL = 0.03


def run_pnp(denoiser, rhos, sigmas, dataset, name):
    x = Variable()
    y = Placeholder()
    mask = Placeholder()
    data_term = csmri(x, mask, y)
    reg_term = deep_prior(x, denoiser='drunet')
    solver = CustomADMM([reg_term], [data_term]).cuda()

    max_iter = len(rhos)

    def step_fn(batch):
        y.value = batch['y0'].cuda()
        mask.value = batch['mask'].cuda()
        target = batch['gt'].cuda()
        x0 = batch['x0'].cuda()
        pred = solver.solve(x0=x0, rhos=rhos, lams={reg_term: sigmas}, max_iter=max_iter).real
        return target, pred

    save_root = f'tmp/pnp_{denoiser}'

    total_psnr = 0
    test_loader = DataLoader(dataset)

    for idx, batch in enumerate(test_loader):
        with torch.no_grad():
            target, pred = step_fn(batch)
        psnr = psnr_qrnn3d(target.squeeze(0).cpu().numpy(),
                           pred.squeeze(0).cpu().numpy(),
                           data_range=1)
        total_psnr += psnr

    print('{} avg psnr= {}'.format(name, total_psnr / len(test_loader)))
    return total_psnr / len(test_loader)

# ---------------------------------------------------------------------------- #
#                                  PnP DRUNET                                  #
# ---------------------------------------------------------------------------- #


def test_pnp_drunet_medical7_4x_n5():
    max_iter = 24
    rhos, sigmas = log_descent(80, 40, max_iter)
    rhos, _ = log_descent(10, 0.1, max_iter)
    psnr = run_pnp(denoiser='drunet', rhos=rhos, sigmas=sigmas,
                   dataset=EvalDataset('data/csmri/Medical7_2020/radial_128_4/5'),
                   name='Medical7_2020/radial_128_4/5')
    assert abs(psnr - 31.78) < TOL


def test_pnp_drunet_medical7_4x_n15():
    max_iter = 24
    rhos, sigmas = log_descent(80, 55, max_iter)
    rhos, _ = log_descent(1, 0.1, max_iter)
    psnr = run_pnp(denoiser='drunet', rhos=rhos, sigmas=sigmas,
                   dataset=EvalDataset('data/csmri/Medical7_2020/radial_128_4/15'),
                   name='Medical7_2020/radial_128_4/15')
    assert abs(psnr - 28.43) < TOL


def test_pnp_drunet_miccai2020_4x_n5():
    max_iter = 24
    rhos, sigmas = log_descent(60, 40, max_iter)
    rhos, _ = log_descent(4, 0.1, max_iter)
    psnr = run_pnp(denoiser='drunet', rhos=rhos, sigmas=sigmas,
                   dataset=EvalDataset('data/csmri/MICCAI_2020/radial_128_4/5'),
                   name='MICCAI_2020/radial_128_4/5')
    assert abs(psnr - 35.57) < TOL


def test_pnp_drunet_miccai2020_8x_n5():
    max_iter = 24
    rhos, sigmas = log_descent(60, 40, max_iter)
    rhos, _ = log_descent(1, 0.1, max_iter)
    psnr = run_pnp(denoiser='drunet', rhos=rhos, sigmas=sigmas,
                   dataset=EvalDataset('data/csmri/MICCAI_2020/radial_128_8/5'),
                   name='MICCAI_2020/radial_128_8/5')
    assert abs(psnr - 32.19) < TOL


# ---------------------------------------------------------------------------- #
#                                   PnP UNet                                   #
# ---------------------------------------------------------------------------- #

def test_pnp_unet_medical7_4x_n5():
    max_iter = 24
    rhos, sigmas = log_descent(70, 40, max_iter)
    rhos, _ = log_descent(120, 0.1, max_iter)
    psnr = run_pnp(denoiser='unet', rhos=rhos, sigmas=sigmas,
                   dataset=EvalDataset('data/csmri/Medical7_2020/radial_128_4/5'),
                   name='Medical7_2020/radial_128_4/5')
    assert abs(psnr - 31.64) < TOL


def test_pnp_unet_medical7_4x_n15():
    max_iter = 24
    rhos, sigmas = log_descent(50, 40, max_iter)
    rhos, _ = log_descent(0.1, 0.1, max_iter)
    psnr = run_pnp(denoiser='unet', rhos=rhos, sigmas=sigmas,
                   dataset=EvalDataset('data/csmri/Medical7_2020/radial_128_4/15'),
                   name='Medical7_2020/radial_128_4/15')
    assert abs(psnr - 28.18) < TOL


def test_pnp_unet_miccai2020_4x_n5():
    max_iter = 24
    rhos, sigmas = log_descent(60, 40, max_iter)
    rhos, _ = log_descent(4, 0.1, max_iter)
    psnr = run_pnp(denoiser='unet', rhos=rhos, sigmas=sigmas,
                   dataset=EvalDataset('data/csmri/MICCAI_2020/radial_128_4/5'),
                   name='MICCAI_2020/radial_128_4/5')
    assert abs(psnr - 35.56) < TOL


def test_pnp_unet_miccai2020_8x_n5():
    max_iter = 24
    rhos, sigmas = log_descent(60, 40, max_iter)
    rhos, _ = log_descent(1, 0.1, max_iter)
    psnr = run_pnp(denoiser='unet', rhos=rhos, sigmas=sigmas,
                   dataset=EvalDataset('data/csmri/MICCAI_2020/radial_128_8/5'),
                   name='MICCAI_2020/radial_128_8/5')
    assert abs(psnr - 32.19) < TOL

# ---------------------------------------------------------------------------- #
#                                      DEQ                                     #
# ---------------------------------------------------------------------------- #


def run_deq(denoiser, dataset, name, ckpt_dir='ckpt/deq_unet2'):
    x = Variable()
    y = Placeholder()
    mask = Placeholder()
    data_term = csmri(x, mask, y)
    reg_term = deep_prior(x, denoiser=denoiser, trainable=True)
    solver = CustomADMM([reg_term], [data_term]).cuda()
    deq_solver = DEQSolver(solver, learned_params=True).cuda()

    def step_fn(batch):
        y.value = batch['y0'].cuda()
        mask.value = batch['mask'].cuda()
        target = batch['gt'].cuda()
        x0 = batch['x0'].cuda()

        max_iter = 1
        rhos, _ = log_descent(1, 1, max_iter, lam=0.5)
        _, sigmas = log_descent(130, 130, max_iter)

        pred = deq_solver.solve(x0=x0, rhos=rhos, lams={reg_term: sigmas}, max_iter=max_iter).real
        return target, pred

    ckpt = torch.load(os.path.join(ckpt_dir, 'last.pth'))
    deq_solver.load_state_dict(ckpt['solver'], strict=True)
    deq_solver.eval()

    total_psnr = 0
    test_loader = DataLoader(dataset)

    for batch in test_loader:
        with torch.no_grad():
            target, pred = step_fn(batch)
        psnr = psnr_qrnn3d(target.squeeze(0).cpu().numpy(),
                           pred.squeeze(0).cpu().numpy(),
                           data_range=1)
        total_psnr += psnr

    print('{} avg psnr= {}'.format(name, total_psnr / len(test_loader)))
    return total_psnr / len(test_loader)


def test_deq_unet_medical7_4x_n5():
    psnr = run_deq(denoiser='unet',
                   dataset=EvalDataset('data/csmri/Medical7_2020/radial_128_4/5'),
                   name='Medical7_2020/radial_128_4/5')
    assert abs(psnr - 31.31) < TOL


def test_deq_unet_medical7_4x_n15():
    psnr = run_deq(denoiser='unet',
                   dataset=EvalDataset('data/csmri/Medical7_2020/radial_128_4/15'),
                   name='Medical7_2020/radial_128_4/15')
    assert abs(psnr - 28.09) < TOL


def test_deq_unet_miccai2020_4x_n5():
    psnr = run_deq(denoiser='unet',
                   dataset=EvalDataset('data/csmri/MICCAI_2020/radial_128_4/5'),
                   name='MICCAI_2020/radial_128_4/5')
    assert abs(psnr - 35.63) < TOL


def test_deq_unet_miccai2020_8x_n5():
    psnr = run_deq(denoiser='unet',
                   dataset=EvalDataset('data/csmri/MICCAI_2020/radial_128_8/5'),
                   name='MICCAI_2020/radial_128_8/5')
    assert abs(psnr - 32.09) < TOL


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


def test_deq_unet_medical7_4x_n15_2():
    x = Variable()
    y = Placeholder()
    mask = Placeholder()
    data_term = csmri(x, mask, y)
    reg_term = deep_prior(x, denoiser='unet', trainable=True, sqrt=False)
    solver = CustomADMM([reg_term], [data_term]).cuda()

    tf_solver = AutoTuneSolver(solver, policy='resnet', action_pack=1, max_episode_step=30,
                               custom_policy_ob_pack_fn=custom_policy_ob_pack_fn)
    ckpt_path = 'ckpt/custom_admm_pack1-train2/actor_best.pkl'
    tf_solver.load(ckpt_path)
    deq_solver = CustomDEQSolver(solver, tf_solver)

    def step_fn(batch):
        y.value = batch['y0'].cuda()
        mask.value = batch['mask'].cuda()
        target = batch['gt'].cuda()
        x0 = batch['x0'].cuda()
        pred = deq_solver.forward(x0, batch).real
        return target, pred

    ckpt = torch.load('ckpt/deq_csmri_admm_iter1/epoch_9.pth')
    deq_solver.load_state_dict(ckpt['solver'], strict=False)
    deq_solver.eval()

    total_psnr = 0
    test_loader = DataLoader(EvalDataset('data/csmri/Medical7_2020/radial_128_4/15'))
    for batch in test_loader:
        with torch.no_grad():
            target, pred = step_fn(batch)
        psnr = psnr_qrnn3d(target.squeeze(0).cpu().numpy(),
                           pred.squeeze(0).cpu().numpy(),
                           data_range=1)
        total_psnr += psnr
    avg_psnr = total_psnr / len(test_loader)
    print('avg psnr=', avg_psnr)
    assert abs(avg_psnr - 28.50) < TOL


# ---------------------------------------------------------------------------- #
#                                  Unroll UNet                                 #
# ---------------------------------------------------------------------------- #


def run_unroll(denoiser, dataset, name, ckpt_dir='ckpt/unroll_unet'):
    x = Variable()
    y = Placeholder()
    mask = Placeholder()
    data_term = csmri(x, mask, y)
    reg_term = deep_prior(x, denoiser=denoiser, trainable=True)
    solver = CustomADMM([reg_term], [data_term])

    max_iter = 10
    solver.rhos = nn.parameter.Parameter(torch.ones([max_iter]))
    solver.sigmas = nn.parameter.Parameter(torch.ones([max_iter]))
    solver = solver.cuda()

    def step_fn(batch):
        y.value = batch['y0'].cuda()
        mask.value = batch['mask'].cuda()
        target = batch['gt'].cuda()
        x0 = batch['x0'].cuda()
        pred = solver.solve(x0=x0, rhos=solver.rhos, lams={reg_term: solver.sigmas}, max_iter=max_iter).real
        return target, pred

    ckpt = torch.load(os.path.join(ckpt_dir, 'last.pth'))
    solver.load_state_dict(ckpt['solver'], strict=True)
    solver.eval()

    total_psnr = 0
    test_loader = DataLoader(dataset)

    for batch in test_loader:
        with torch.no_grad():
            target, pred = step_fn(batch)
        psnr = psnr_qrnn3d(target.squeeze(0).cpu().numpy(),
                           pred.squeeze(0).cpu().numpy(),
                           data_range=1)
        total_psnr += psnr
    print('{} avg psnr= {}'.format(name, total_psnr / len(test_loader)))
    return total_psnr / len(test_loader)


def test_unroll_unet_medical7_4x_n5():
    psnr = run_unroll(denoiser='unet',
                      dataset=EvalDataset('data/csmri/Medical7_2020/radial_128_4/5'),
                      name='Medical7_2020/radial_128_4/5')
    assert abs(psnr - 31.86) < TOL


def test_unroll_unet_medical7_4x_n15():
    psnr = run_unroll(denoiser='unet',
                      dataset=EvalDataset('data/csmri/Medical7_2020/radial_128_4/15'),
                      name='Medical7_2020/radial_128_4/15')
    assert abs(psnr - 28.66) < TOL


def test_unroll_unet_miccai2020_4x_n5():
    psnr = run_unroll(denoiser='unet',
                      dataset=EvalDataset('data/csmri/MICCAI_2020/radial_128_4/5'),
                      name='MICCAI_2020/radial_128_4/5')
    assert abs(psnr - 36.25) < TOL


def test_unroll_unet_miccai2020_8x_n5():
    psnr = run_unroll(denoiser='unet',
                      dataset=EvalDataset('data/csmri/MICCAI_2020/radial_128_8/5'),
                      name='MICCAI_2020/radial_128_8/5')
    assert abs(psnr - 32.56) < TOL

# ---------------------------------------------------------------------------- #
#                                  RL UNet                                     #
# ---------------------------------------------------------------------------- #


def run_tfpnp(denoiser, dataset, name):
    x = Variable()
    y = Placeholder()
    mask = Placeholder()
    data_term = csmri(x, mask, y)
    reg_term = deep_prior(x, denoiser=denoiser)
    solver = CustomADMM([reg_term], [data_term]).cuda()

    tf_solver = AutoTuneSolver(solver, policy='resnet', custom_policy_ob_pack_fn=custom_policy_ob_pack_fn)

    def step_fn(batch):
        y.value = batch['y0'].cuda()
        mask.value = batch['mask'].cuda()
        target = batch['gt'].cuda()
        x0 = batch['x0'].cuda()
        pred = tf_solver.solve(x0, batch).real
        return target, pred

    ckpt_path = 'ckpt/tfpnp_unet/actor_best.pkl'
    tf_solver.load(ckpt_path)

    total_psnr = 0
    test_loader = DataLoader(dataset)

    for idx, batch in enumerate(test_loader):
        with torch.no_grad():
            target, pred = step_fn(batch)
        psnr = psnr_qrnn3d(target.squeeze(0).cpu().numpy(),
                           pred.squeeze(0).cpu().numpy(),
                           data_range=1)
        total_psnr += psnr

    avg_psnr = total_psnr / len(test_loader)
    print('{} avg psnr= {}'.format(name, avg_psnr))
    return avg_psnr


def test_tfpnp_unet_medical7_4x_n5():
    psnr = run_tfpnp(denoiser='unet',
                     dataset=EvalDataset('data/csmri/Medical7_2020/radial_128_4/5'),
                     name='Medical7_2020/radial_128_4/5')
    assert abs(psnr - 31.82) < TOL


def test_tfpnp_unet_medical7_4x_n15():
    psnr = run_tfpnp(denoiser='unet',
                     dataset=EvalDataset('data/csmri/Medical7_2020/radial_128_4/15'),
                     name='Medical7_2020/radial_128_4/15')
    assert abs(psnr - 28.57) < TOL


def test_tfpnp_unet_miccai2020_4x_n5():
    psnr = run_tfpnp(denoiser='unet',
                     dataset=EvalDataset('data/csmri/MICCAI_2020/radial_128_4/5'),
                     name='MICCAI_2020/radial_128_4/5')
    assert abs(psnr - 36.21) < TOL


def test_tfpnp_unet_miccai2020_8x_n5():
    psnr = run_tfpnp(denoiser='unet',
                     dataset=EvalDataset('data/csmri/MICCAI_2020/radial_128_8/5'),
                     name='MICCAI_2020/radial_128_8/5')
    assert abs(psnr - 32.70) < TOL

# ---------------------------------------------------------------------------- #
#                                   RL DRUNet                                  #
# ---------------------------------------------------------------------------- #


def test_tfpnp_drunet_medical7_4x_n5():
    psnr = run_tfpnp(denoiser='drunet',
                     dataset=EvalDataset('data/csmri/Medical7_2020/radial_128_4/5'),
                     name='Medical7_2020/radial_128_4/5')
    assert abs(psnr - 32.66) < TOL


def test_tfpnp_drunet_medical7_4x_n15():
    psnr = run_tfpnp(denoiser='drunet',
                     dataset=EvalDataset('data/csmri/Medical7_2020/radial_128_4/15'),
                     name='Medical7_2020/radial_128_4/15')
    assert abs(psnr - 28.91) < TOL


def test_tfpnp_drunet_miccai2020_4x_n5():
    psnr = run_tfpnp(denoiser='drunet',
                     dataset=EvalDataset('data/csmri/MICCAI_2020/radial_128_4/5'),
                     name='MICCAI_2020/radial_128_4/5')
    assert abs(psnr - 36.64) < TOL


def test_tfpnp_drunet_miccai2020_8x_n5():
    psnr = run_tfpnp(denoiser='drunet',
                     dataset=EvalDataset('data/csmri/MICCAI_2020/radial_128_8/5'),
                     name='MICCAI_2020/radial_128_8/5')
    assert abs(psnr - 33.22) < TOL
