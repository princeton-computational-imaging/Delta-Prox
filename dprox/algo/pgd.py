from typing import List

from dprox.proxfn import ProxFn

from .base import Algorithm, expand


class ProximalGradientDescent(Algorithm):
    @classmethod
    def partition(cls, prox_fns: List[ProxFn]):
        if len(prox_fns) != 2:
            raise ValueError('Proximal gradient descent only supports \
                two proximal functions for now.')

        omega_fns = []
        for fn in prox_fns:
            if hasattr(fn, 'grad'):
                omega_fns.append(fn)

        psi_fns = [fn for fn in prox_fns if fn not in omega_fns]

        if len(omega_fns) == 0:
            raise ValueError('Proximal gradient descent requires \
                at least one proximal function is differentiable.')

        return psi_fns, omega_fns

    def __init__(
        self,
        psi_fns: List[ProxFn],
        omega_fns: List[ProxFn],
        *args,
        **kwargs
    ):
        super().__init__(psi_fns, omega_fns)
        self.diff_fn = omega_fns[0]
        self.prox_fn = psi_fns[0]

    def _iter(self, state, rho, lam):
        x = state[0]
        v = x - expand(rho) * self.diff_fn.grad(x)
        x = self.prox_fn.prox(v, lam[self.prox_fn])
        return [x]

    def initialize(self, x0):
        return [x0]

    @property
    def state_split(self):
        return [1]

    @property
    def nparams(self):
        return len(self.psi_fns) + 1
