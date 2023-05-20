from ..linop import adjoint
from .admm import ADMM
from .base import expand


class PockChambolle(ADMM):
    def initialize(self, x0):
        x = x0
        xbar = x.clone()
        z = self.K.forward(x, return_list=True)
        return x, z, xbar

    def _iter(self, state, rho, lam):
        x, z, xbar = state

        # update z
        Kxbar = self.K.forward(xbar, return_list=True)
        for i, fn in enumerate(self.psi_fns):
            r = expand(lam[fn])
            z[i] = z[i] + r * Kxbar[i]
            z[i] = z[i] - r * fn.prox(z[i], lam=r)

        # update x
        # Ktz = self.K.adjoint(z)
        Ktz = [adjoint(fn.linop, z[i]) for i, fn in enumerate(self.psi_fns)]
        x_next = [x - Ktz[i] for i in range(len(Ktz))]
        if len(self.omega_fns) > 0:
            x_next = self.least_square.solve(x_next, rho)
        else:
            x_next = sum(x_next)

        # update xbar
        xbar = x_next + x_next - x
        x = x_next

        return x, z, xbar

    @property
    def state_split(self):
        return [1, [len(self.psi_fns)], 1]
