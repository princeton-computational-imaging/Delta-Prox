from .nlm import NonLocalMeansFast
from .. import ProxFn


class patch_nlm(ProxFn):
    def __init__(self, linop):
        super().__init__(linop)
        self.denoiser = NonLocalMeansFast()

    def _prox(self, v, lam):
        lam = lam.sqrt()
        out = self.denoiser(v, lam)
        return out
