from typing import List, Union

import torch

from dprox.proxfn import ProxFn

from . import opt
from .admm import ADMM, ADMM_vxu, LinearizedADMM
from .base import Algorithm
from .hqs import HQS
from .pc import PockChambolle
from .pgd import ProximalGradientDescent
from .special import DEQSolver

SOLVERS = {
    'admm': ADMM,
    'admm_vxu': ADMM_vxu,
    'ladmm': LinearizedADMM,
    'hqs': HQS,
    'pc': PockChambolle,
    'pgd': ProximalGradientDescent,
}

SPECAILIZATIONS = {
    'deq': DEQSolver
}


def compile(prox_fns, method='admm', device='cuda', **kwargs):
    algorithm: Algorithm = SOLVERS[method]
    device = torch.device(device) if isinstance(device, str) else device

    psi_fns, omega_fns = algorithm.partition(prox_fns)
    solver = algorithm.create(psi_fns, omega_fns, **kwargs)
    solver = solver.to(device)
    return solver


def specialize(solver, method='deq', **kwargs):
    return SPECAILIZATIONS[method](solver, **kwargs)


def optimize(prox_fns, merge=False, absorb=False):
    if absorb:
        prox_fns = opt.absorb.absorb_all_linops(prox_fns)
    return prox_fns


def visualize():
    pass


class Problem:
    def __init__(
        self,
        prox_fns: Union[ProxFn, List[ProxFn]],
        absorb=True,
        merge=True,
        try_diagonalize=True,
        try_freq_diagonalize=True,
        lin_solver_kwargs={},
    ):
        if isinstance(prox_fns, ProxFn):
            prox_fns = [prox_fns]
        self.prox_fns = prox_fns

        self.absorb = absorb
        self.merge = merge

        self.solver_args = dict(
            try_diagonalize=try_diagonalize,
            try_freq_diagonalize=try_freq_diagonalize,
            lin_solver_kwargs=lin_solver_kwargs,
        )

    @property
    def objective(self):
        return self.prox_fns

    def solve(self, method='admm', device='cuda', **kwargs):
        prox_fns = optimize(self.prox_fns, merge=self.merge, absorb=self.absorb)
        print(prox_fns)
        solver = compile(prox_fns, method=method, device=device, **self.solver_args)
        results = solver.solve(**kwargs)
        return results

    def visuliaze(self, savepath):
        pass
