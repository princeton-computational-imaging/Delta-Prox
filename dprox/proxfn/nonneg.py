import torch

from . import ProxFn


class nonneg(ProxFn):
    def __init__(self, linop=None):
        super().__init__(linop)

    def _prox(self, v, lam):
        return torch.maximum(v, torch.zeros_like(v))
