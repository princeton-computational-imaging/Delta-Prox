import torch

from .. import ProxFn
from .dgu import Denoiser


class unrolled_prior(ProxFn):
    def __init__(self, linop, denoiser=None):
        super().__init__(linop)
        if denoiser is not None:
            self.denoiser = denoiser()
        else:
            self.denoiser = Denoiser()
    
    def eval(self, v):
        raise NotImplementedError('deep prior cannot be explictly evaluated')

    def _prox(self, v: torch.Tensor, lam: torch.Tensor=None):
        """ v: [N, C, H, W]
            lam: [1]
        """
        out = self.denoiser(v, self.step)
        return out