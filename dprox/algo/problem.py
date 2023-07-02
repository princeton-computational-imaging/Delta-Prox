from typing import List, Union

import torch

from dprox.linalg import LinearSolveConfig
from dprox.linop.constaints import equality, less, matmul
from dprox.proxfn import ProxFn

from . import lp, opt
from .admm import ADMM, ADMM_vxu, LinearizedADMM
from .base import Algorithm
from .hqs import HQS
from .pc import PockChambolle
from .pgd import ProximalGradientDescent
from .special import DEQSolver, UnrolledSolver

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
    'rl': UnrolledSolver,
    'unroll': UnrolledSolver,
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


class Problem:
    def __init__(
        self,
        prox_fns: Union[ProxFn, List[ProxFn]],
        constraints=[],
        absorb=True,
        merge=True,
        try_diagonalize=True,
        try_freq_diagonalize=True,
        linear_solve_config=LinearSolveConfig(),
    ):
        self.prob = None
        if isinstance(prox_fns, matmul):
            self.prob = LPProblem(prox_fns, constraints)
            return

        if isinstance(prox_fns, ProxFn):
            prox_fns = [prox_fns]
        self.prox_fns = prox_fns

        self.absorb = absorb
        self.merge = merge

        self.solver_args = dict(
            try_diagonalize=try_diagonalize,
            try_freq_diagonalize=try_freq_diagonalize,
            linear_solve_config=linear_solve_config,
        )

    @property
    def objective(self):
        return self.prox_fns

    def solve(self, method='admm', device='cuda', **kwargs):
        if self.prob is not None:
            return self.prob.solve()

        # TODO: optimize has bug when scale is absorbed in test_ml_problems
        # prox_fns = optimize(self.prox_fns, merge=self.merge, absorb=self.absorb)
        prox_fns = self.prox_fns
        solver = compile(prox_fns, method=method, device=device, **self.solver_args)
        results = solver.solve(**kwargs)
        return results

    def visuliaze(self, savepath):
        pass


class LPProblem:
    def __init__(
        self,
        objective: matmul,
        constraints,
        max_iters=20000,
        abstol=1e-3,
        reltol=1e-6,
        rho=1e-1,
        device=torch.device('cuda')
    ):
        self.objective = objective
        self.constraints = constraints

        norm_ord = float('inf')
        dtype = torch.float64

        c = objective.A
        assert len(constraints) == 2
        for constraint in constraints:
            if isinstance(constraint, equality):
                A_eq = constraint.left.A
                b_eq = constraint.right
            if isinstance(constraint, less):
                A_ub = constraint.left.A
                b_ub = constraint.right

        self.prob = lp.LPProblem(c, A_ub, b_ub, A_eq, b_eq, norm_ord=norm_ord, dtype=dtype, sparse=True, device=device)
        self.solver = lp.LPSolverADMM(rho=rho, problem_scale=None, abstol=abstol, reltol=reltol, max_iters=max_iters, dtype=dtype).to(device)

    def optimize_params(self):
        base_lr = 5e-3
        optimizer = torch.optim.Adam(self.solver.parameters(), lr=base_lr)
        criterion = lp.LPConvergenceLoss()

        loss_log = []
        num_iters = 10

        for k in range(num_iters):
            # adjust_lr_cosine(optimizer, k, num_iters, base_lr=base_lr, min_lr=1e-3)
            optimizer.zero_grad()
            _, _, res = self.solver.solve(self.prob, max_iters=10)
            objval, r_norm, s_norm, eps_primal, eps_dual = res

            # define loss
            loss = criterion(r_norm, s_norm, eps_primal, eps_dual)
            loss.backward()
            optimizer.step()

            loss_log.append(loss.item())

            print(loss.item())
            print(self.solver)

    def solve(self, method='admm', adapt_params=True):
        self.optimize_params()
        with torch.no_grad():
            self.solver.eval()
            x, history, res = self.solver.solve(self.prob, residual_balance=True)
        return x.min()
