import torch.nn as nn
import copy
import torch

from ..base import move, auto_convert_to_tensor

def clone(x, nums, shared):
    return [x if shared else copy.deepcopy(x) for _ in range(nums)]


class UnrolledSolver(nn.Module):
    def __init__(self, solver, max_iter, shared=False, learned_params=False):
        super().__init__()
        if shared==False:   
            self.solvers = nn.ModuleList(clone(solver, max_iter, shared=shared))
        else:
            self.solver = solver
            self.solvers = [self.solver for _ in range(max_iter)]
            
        self.max_iter = max_iter
        self.shared = shared
        
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
            rho = rhos[..., i:i+1]
            lam = {self.solvers[i].psi_fns[0]: v[..., i:i+1] for k, v in lams.items()}
            state = self.solvers[i].iters(state, rho, lam, 1, False)
        
        return state[0]