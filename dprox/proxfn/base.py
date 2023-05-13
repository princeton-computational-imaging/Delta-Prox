import numpy as np

from dprox.linop import LinOp, Placeholder, CompGraph
from dprox.utils import to_torch_tensor
import torch.nn as nn


def exists(x):
    return x is not None


def prox_scaled(prox, alpha):
    def _prox(v, lam):
        return prox(v, lam * alpha)
    return _prox


def prox_affine(prox, beta):
    def _prox(v, lam):
        return 1. / beta * prox(beta * v, beta * beta * lam)
    return _prox


def prox_translated(prox, b):
    def _prox(v, lam):
        return prox(v - b, lam) + b
    return _prox


class ProxFn(nn.Module):
    """ The definition of proximal operator is
        f(x) = argmin_x f(x) + 1/(2*lam) * ||x-v||_2^2 
    """

    def __init__(self, linop: LinOp, alpha=1, beta=1):
        super().__init__()
        self.linop = linop
        self.alpha = alpha
        self.beta = beta
        self.step = 0
        self.dag = CompGraph(linop, zero_out_constant=True)

    @property
    def offset(self):
        return -self.linop.offset

    def unwrap(self, value):
        if isinstance(value, Placeholder):
            return value.value
        return to_torch_tensor(value, batch=True).to(self.linop.device)

    def eval(self, v):
        return NotImplementedError

    def prox(self, v, lam):
        """ v: [B,C,H,W], lam: [B]
        """
        if len(lam.shape) == 1: lam = lam.view(lam.shape[0], 1, 1, 1)

        fn = self._prox
        fn = prox_scaled(fn, self.alpha)
        fn = prox_affine(fn, self.beta)
        fn = prox_translated(fn, self.offset)
        return fn(v, lam)

    def convex_conjugate_prox(self, v, lam):
        # use Moreau’s identity
        return v - self.prox(v / lam, lam)

    def _prox(self, v, lam):
        return NotImplementedError

    # def grad(self, x):
    #     x_ = x.detach().requires_grad_(True)
    #     self.eval(x_).backward()
    #     return x_.grad

    def __mul__(self, other):
        if np.isscalar(other) and other > 0:
            self.alpha = other
            return self
        return TypeError("Can only multiply by a positive scalar.")

    def __rmul__(self, other):
        """Called for Number * ProxFn.
        """
        return self * other

    def __add__(self, other):
        """ProxFn + ProxFn(s).
        """
        if isinstance(other, ProxFn):
            return [self, other]
        elif type(other) == list:
            return [self] + other
        else:
            return NotImplemented

    def __radd__(self, other):
        """Called for list + ProxFn.
        """
        if type(other) == list:
            return other + [self]
        else:
            return NotImplemented

    def __str__(self):
        return f'{self.__class__.__name__}'
