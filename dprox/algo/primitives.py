from pathlib import Path
from typing import List, Union

import torch
import torch.nn.functional as F
import torchlight as tl
import torchlight.nn as tlnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dprox import *
from dprox.contrib.optic import Dataset
from dprox.proxfn import ProxFn
from dprox.utils import *

from . import opt
from .admm import ADMM, ADMM_vxu, LinearizedADMM
from .base import Algorithm
from .hqs import HQS
from .pc import PockChambolle
from .pgd import ProximalGradientDescent
from .specialization import DEQSolver, UnrolledSolver, AutoTuneSolver, build_unrolled_solver

SOLVERS = {
    'admm': ADMM,
    'admm_vxu': ADMM_vxu,
    'ladmm': LinearizedADMM,
    'hqs': HQS,
    'pc': PockChambolle,
    'pgd': ProximalGradientDescent,
}

SPECAILIZATIONS = {
    'deq': DEQSolver,
    'rl': AutoTuneSolver,
    'unroll': build_unrolled_solver,
}

def compile(
    prox_fns: List[ProxFn],
    method: str = 'admm',
    device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
    **kwargs
):
    """
    Compile the given objective (in terms of a list of proxable functions) into a proximal solver.

    >>> solver = compile(data_term+reg_term, method='admm')

    Args:
      prox_fns (List[ProxFn]): A list or the sum of proxable functions. 
      method (str): A string that specifies the name of the optimization method to use. Defaults to `admm`.
        Valid methods include [`admm`, `admm_vxu`, `ladmm`, `hqs`, `pc`, `pgd`]. 
      device (Union[str, torch.device]): The device (CPU or GPU) on which the solver should run. 
        It can be either a string ('cpu' or 'cuda') or a `torch.device` object. Defaults to cuda if avaliable.

    Returns:
      An instance of a solver object that is created using the specified algorithm and proximal functions. 
    """
    algorithm: Algorithm = SOLVERS[method]
    device = torch.device(device) if isinstance(device, str) else device

    psi_fns, omega_fns = algorithm.partition(prox_fns)
    solver = algorithm.create(psi_fns, omega_fns, **kwargs)
    solver = solver.to(device)
    return solver


def specialize(
    solver: Algorithm,
    method: str = 'deq',
    device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
    **kwargs
):
    """ 
    Specialize the given solver based on the given method. 

    >>> deq_solver = specialize(solver, method='deq')
    >>> rl_solver = specialize(solver, method='rl')
    >>> unroll_solver = specialize(solver, method='unroll')

    Args:
      solver (Algorithm): the proximal solver that need to be specialized.
      method (str): the strategy for the specialization. Choose from [`deq`, `rl`, `unroll`].
      device (Union[str, torch.device]): The device (CPU or GPU) on which the solver should run. 
        It can be either a string ('cpu' or 'cuda') or a `torch.device` object. Defaults to cuda if avaliable

    Returns:
      The specialized solver.
    """
    solver = SPECAILIZATIONS[method](solver, **kwargs)
    device = torch.device(device) if isinstance(device, str) else device
    solver = solver.to(device)
    return solver


def optimize(
    prox_fns: List[ProxFn],
    merge=False,
    absorb=False
):
    if absorb:
        prox_fns = opt.absorb.absorb_all_linops(prox_fns)
    return prox_fns


def visualize():
    pass


def train(
    model,
    step_fn,
    dataset='BSD500',
    savedir='saved',
    epochs=10,
    bs=2,
    lr=1e-4,
    resume=None,
):
    savedir = Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    logger = tl.logging.Logger(savedir)

    # ----------------- Start Training ------------------------ #
    root = hf.download_dataset(dataset, force_download=False)
    dataset = Dataset(root)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4, weight_decay=1e-3
    )
    tlnn.utils.adjust_learning_rate(optimizer, lr)

    epoch = 0
    gstep = 0
    best_psnr = 0
    imgdir = savedir / 'imgs'
    imgdir.mkdir(exist_ok=True, parents=True)

    if resume:
        ckpt = torch.load(savedir / resume)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        epoch = ckpt['epoch'] + 1
        gstep = ckpt['gstep'] + 1
        best_psnr = ckpt['best_psnr']

    def save_ckpt(name, psnr=0):
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'gstep': gstep,
            'psnr': psnr,
            'best_psnr': best_psnr,
        }
        torch.save(ckpt, savedir / name)

    save_ckpt('last.pth')
    while epoch < epochs:
        tracker = tl.trainer.util.MetricTracker()
        pbar = tqdm(total=len(loader), dynamic_ncols=True, desc=f'Epcoh[{epoch}]')

        for i, batch in enumerate(loader):

            gt, inp, pred = step_fn(batch)

            loss = F.mse_loss(gt, pred)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            psnr = tl.metrics.psnr(pred, gt)
            loss = loss.item()
            tracker.update('loss', loss)
            tracker.update('psnr', psnr)

            pbar.set_postfix({'loss': f'{tracker["loss"]:.4f}',
                              'psnr': f'{tracker["psnr"]:.4f}'})
            pbar.update()

            gstep += 1

        logger.info('Epoch {} Loss={} LR={}'.format(epoch, tracker['loss'], tlnn.utils.get_learning_rate(optimizer)))

        save_ckpt('last.pth', tracker['psnr'])
        pbar.close()
        epoch += 1
