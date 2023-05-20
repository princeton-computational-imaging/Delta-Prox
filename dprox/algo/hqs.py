from .admm import ADMM


class HQS(ADMM):
    def initialize(self, x0):
        x = x0
        z = self.K.forward(x, return_list=True)
        return x, z

    def _iter(self, state, rho, lam):
        x, z = state
        x = self.least_square.solve(z, rho)
        Kx = self.K.forward(x, return_list=True)  # cache Kx
        for i, fn in enumerate(self.psi_fns):
            z[i] = fn.prox(Kx[i], lam=lam[fn])
        return x, z

    @property
    def state_split(self):
        return [1, [len(self.psi_fns)]]
