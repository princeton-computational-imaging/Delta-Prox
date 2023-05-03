from typing import List

import torch

from dprox.linop import LinOp, vstack, eval, adjoint
from .linear_solve import LINEAR_SOLVER

from . import ProxFn


class sum_squares(ProxFn):
    """ |x|_2^2 
    """

    def __init__(self, linop, b=None, eps=1e-7):
        super().__init__(linop)
        self.eps = eps
        self._b = b

    @property
    def b(self):
        if self._b is not None:
            return self.unwrap(self._b)
        return super().b

    def _prox(self, v, lam):
        return v / (1 + 2 * lam)

    def grad(self, x):
        tmp = eval(self.linop, [x])[0] - self.b
        out = adjoint(self.linop, [tmp])[0]
        return out


class ext_sum_squares(sum_squares):
    def __init__(self, linop, eps=1e-7):
        super().__init__(linop, eps=eps)

    def setup(self, b):
        # TODO: this might cause bugs, b might not be initialized
        self.quad_b = b
        return self

    def solve(self, b, rho, eps=1e-6):
        xtilde = 0
        for v in b: xtilde += v
        return self._prox(xtilde, rho, len(b))


class weighted_sum_squares(sum_squares):
    """ |Ax-b|_2^2 
        (AtA + I)^-1(Atb + v)
    """

    def __init__(self, linop, weight: LinOp, b, eps=0):
        super().__init__(linop, b, eps)
        self.weight = weight
        if self.weight.is_self_diag():
            self._prox_fn = self._prox
        elif self.weight.is_self_diag(freq=True):
            self._prox_fn = self._prox_freq
        else:
            raise ValueError('weight {} must be diagonalizable'.format(weight))

    @property
    def Ktb(self):
        return adjoint(self.weight, [self.unwrap(self._b)])[0]

    def prox(self, v, lam):
        return self._prox_fn(v, lam)

    def _prox(self, v, lam):
        if len(lam.shape) == 1:
            lam = lam.view(lam.shape[0], 1, 1, 1)
        Ktb = self.Ktb.to(v.device)
        diag = self.weight.get_diag(Ktb).to(v.device)
        return (Ktb + lam * v) / (diag + lam)

    def _prox_freq(self, v, lam):
        Ktb = torch.fft.fftn(self.Ktb + lam * v, dim=[0, 1])
        fout = (Ktb + self.eps) / (self.weight.get_diag(v, freq=True) + lam + self.eps)
        return torch.real(torch.fft.ifftn(fout, dim=[0, 1]))


class least_squares(ProxFn):
    def __init__(
        self,
        quad_fns: List[ProxFn],
        other_fns: List[ProxFn],
        try_diagonalize=True,
        try_freq_diagonalize=True,
        fallback_solver='cg',
        lin_solver_kwargs={},
    ):
        self.quad_fns = quad_fns
        self.other_fns = other_fns

        self.try_freq_diagonalize = try_freq_diagonalize
        self.try_diagonalize = try_diagonalize
        self.fallback_solver = fallback_solver
        self.lin_solver_kwargs = lin_solver_kwargs

        linops = [fn.linop for fn in quad_fns + other_fns]
        stacked = vstack(linops)
        self.diagonalizable = stacked.is_gram_diag(freq=False) and try_diagonalize
        self.freq_diagonalizable = stacked.is_gram_diag(freq=True) and try_diagonalize and try_freq_diagonalize

        super().__init__(stacked)
        # print(f'diagonalizable={self.diagonalizable}, freq_diagonalizable={self.freq_diagonalizable}')

    def _prox(self, v, lam):
        return self.solve([], lam, v=v)

    def solve(self, b, rho, v=None, eps=1e-7):
        if rho.ndim == 1: rho = rho.view(rho.shape[0], 1, 1, 1)
        if self.diagonalizable or self.freq_diagonalizable:
            return self.solve_direct(b, rho, v, eps)
        else:
            return self.solve_cg(b, rho, v, **self.lin_solver_kwargs)

    def solve_direct(self, b, rho, v=None, eps=1e-7):
        device = rho.device
        Ktb = 0
        for fn in self.quad_fns:
            Ktb += fn.dag.adjoint([fn.b])[0]
        for i, fn in enumerate(self.other_fns):
            Ktb += rho * fn.dag.adjoint([b[i]])[0]
        if v is not None:
            Ktb += rho * v
        
        def get_diag(fn: ProxFn):
            # TODO: hack for derain, don't forgot change it back, Ktb -> Ktb.shape
            return fn.linop.get_diag(Ktb, freq=self.freq_diagonalizable)

        diag = 0
        for fn in self.quad_fns:
            diag = diag + get_diag(fn).to(device)
        for fn in self.other_fns:
            diag = diag + rho * get_diag(fn).to(device)
        if v is not None:
            diag = diag + rho
        
        if self.freq_diagonalizable:
            Ktb = torch.fft.fftn(Ktb, dim=[-2, -1])
            out = torch.real(torch.fft.ifftn((Ktb + eps) / (diag + eps), dim=[-2, -1]))
        else:
            out = Ktb / (diag + eps)

        return out.float()

    def solve_cg(self, b, rho, v=None, **kwargs):
        # KtKfun being a function that computes the matrix vector product KtK x

        def KtK(x):
            out = 0
            for fn in self.quad_fns:
                out += fn.dag.adjoint(fn.dag.forward([x]))[0]
            for fn in self.other_fns:
                out += rho * fn.dag.adjoint(fn.dag.forward([x]))[0]  # slow when rho is small
            if v is not None:
                out += rho * x
            return out

        Ktb = 0
        for fn in self.quad_fns:
            Ktb += fn.dag.adjoint([fn.b])[0]
        for i, fn in enumerate(self.other_fns):
            Ktb += rho * fn.dag.adjoint([b[i]])[0]
        if v is not None:
            Ktb += rho * v
            
        init_with_last_pred = kwargs.pop('init_with_last_pred', False)
        lin_solver_type = kwargs.pop('lin_solver_type', 'cg2')
        lin_solver = LINEAR_SOLVER[lin_solver_type]
        if init_with_last_pred:
            x_init = self.last_x_pred if hasattr(self, 'last_x_pred') else None
            x_pred = lin_solver(KtK, Ktb, x_init=x_init, **kwargs)
            self.last_x_pred = x_pred
        else:
            x_pred = lin_solver(KtK, Ktb, **kwargs)
        return x_pred