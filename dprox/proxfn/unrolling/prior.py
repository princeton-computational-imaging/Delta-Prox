import torch

from .. import ProxFn
from .dgu import Denoiser


class unrolled_prior(ProxFn):
    def __init__(self, linop):
        super().__init__(linop)
        self.denoiser = Denoiser()
    
    def eval(self, v):
        raise NotImplementedError('deep prior cannot be explictly evaluated')

    def _prox(self, v: torch.Tensor, lam: torch.Tensor=None):
        """ v: [N, C, H, W]
            lam: [1]
        """
        out = self.denoiser(v, self.step)
        return out