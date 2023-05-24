from typing import List

import torch

from dprox.proxfn import ProxFn, sum_squares, ext_sum_squares
from dprox.linop import eval, adjoint
from dprox.linalg import LinearSolveConfig

from .base import Algorithm, expand
from .invert import get_least_square_solver


""" 
Given a a list of proxable function
1. variable splitting.
    - ADMM use two groups, one group shares a variable x, all fn in another group use independent variable u.
2. choose proper linear solver. 
    - The update of x relies on a linear solver.
    - The update of u/s rely on their proximal operators.
    - The linear solver can be direct method, if all proxfns' linops are diagonalizable or frequency diagonalizable.
        Otherwise, iterative methods will be used.
"""


class ADMM(Algorithm):
    @classmethod
    def partition(cls, prox_fns: List[ProxFn]):
        omega_fns, flag = [], False  # we only add one extensible sum square to omega
        for fn in prox_fns:
            if not flag and isinstance(fn, ext_sum_squares):
                omega_fns += [fn]
            elif type(fn) == sum_squares and not isinstance(fn, ext_sum_squares):
                omega_fns = [fn]
        psi_fns = [fn for fn in prox_fns if fn not in omega_fns]
        return psi_fns, omega_fns

    def __init__(
        self,
        psi_fns: List[ProxFn],
        omega_fns: List[ProxFn],
        try_diagonalize=True,
        try_freq_diagonalize=True,
        linear_solve_config=LinearSolveConfig()
    ):
        super().__init__(psi_fns, omega_fns)
        self.least_square = get_least_square_solver(psi_fns, omega_fns, try_diagonalize, try_freq_diagonalize, linear_solve_config)

    def _iter(self, state, rho, lam):
        x, v, u = state
        b = [v[i] - u[i] for i in range(len(self.psi_fns))]
        x = self.least_square.solve(b, rho)

        Kx = self.K.forward(x, return_list=True)  # cache Kx
        for i, fn in enumerate(self.psi_fns):
            v[i] = fn.prox(Kx[i] + u[i], lam=lam[fn])
            u[i] = u[i] + Kx[i] - v[i]

        return x, v, u

    def initialize(self, x0):
        x = x0
        v = self.K.forward(x, return_list=True)
        if v is None: v = []  # in case there is no psi fns
        u = [torch.zeros_like(e) for e in v]
        return x, v, u

    @property
    def nparams(self):
        return len(self.psi_fns) + 1

    @property
    def state_split(self):
        return [1, [len(self.psi_fns)], [len(self.psi_fns)]]


class LinearizedADMM(ADMM):
    def _iter(self, state, rho, lam):
        x, v, u = state

        # solve x problem using least square
        b = []
        for i, fn in enumerate(self.psi_fns):
            # coeff = expand(rho/lam[fn])
            # we empircally find ignore the coefficient works better.
            coeff = 1
            tmp = eval(fn.linop, x) - v[i] + u[i]
            tmp = adjoint(fn.linop, tmp)
            b += [x - coeff * tmp]

        x = self.least_square.solve(b, rho)

        # solve proximal fns
        Kx = self.K.forward(x)  # cache Kx
        for i, fn in enumerate(self.psi_fns):
            v[i] = fn.prox(Kx[i] + u[i], lam=lam[fn])
            u[i] = u[i] + Kx[i] - v[i]

        return x, v, u


class ADMM_vxu(ADMM):
    """ update x,v,u in v,x,u order
    """

    def _iter(self, state, rho, lam):
        z, x, u = state

        Kz = self.K.forward(z)  # cache Kx
        for i, fn in enumerate(self.psi_fns):
            x[i] = fn.prox(Kz[i] - u[i], lam=lam[fn])

        b = [x[i] + u[i] for i in range(len(self.psi_fns))]
        z = self.least_square.solve(b, rho)

        for i, fn in enumerate(self.psi_fns):
            u[i] = u[i] + x[i] - z

        return z, x, u
