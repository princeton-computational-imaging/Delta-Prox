import copy
from functools import partial

import torch
import torch.nn as nn

from ..base import auto_convert_to_tensor, move


def clone(x, nums, share):
    return [x if share else copy.deepcopy(x) for _ in range(nums)]


def build_unrolled_solver(solver, share=True, **kwargs):
    if share == True:
        solver.solve = partial(solver.solve, **kwargs)
        return solver
    return UnrolledSolver(solver, share=share, **kwargs)


class UnrolledSolver(nn.Module):
    def __init__(self, solver, max_iter, share=False, learned_params=False):
        super().__init__()
        if share == False:
            self.solvers = nn.ModuleList(clone(solver, max_iter, share=share))
        else:
            self.solver = solver
            self.solvers = [self.solver for _ in range(max_iter)]

        self.max_iter = max_iter
        self.share = share

        self.learned_params = learned_params
        if learned_params:
            self.rhos = nn.parameter.Parameter(torch.ones(max_iter))
            self.lams = {}
            for fn in solver.psi_fns:
                lam = nn.parameter.Parameter(torch.ones(max_iter))
                setattr(self, str(fn), lam)
                self.lams[fn] = lam

    @auto_convert_to_tensor(['x0', 'rhos', 'lams'], batchify=['x0'])
    def solve(self, x0=None, rhos=None, lams=None, max_iter=None):
        x0, rhos, lams = move(x0, rhos, lams, device=self.solvers[0].device)

        if self.learned_params:
            rhos, lams = self.rhos, self.lams

        max_iter = self.max_iter if max_iter is None else max_iter

        state = self.solvers[0].initialize(x0)

        for i in range(max_iter):
            rho = rhos[..., i:i + 1]
            lam = {self.solvers[i].psi_fns[0]: v[..., i:i + 1] for k, v in lams.items()}
            state = self.solvers[i].iters(state, rho, lam, 1, False)

        return state[0]
