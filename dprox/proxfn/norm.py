import torch

from . import ProxFn


def soft_threshold(v, lam):
    """ ref: https://www.tensorflow.org/probability/api_docs/python/tfp/math/soft_threshold

        argmin_x lam * |x|_1 + 0.5 * (x-v)^2
    """
    return torch.sign(v) * torch.maximum(torch.abs(v) - lam, torch.zeros_like(v))


class norm1(ProxFn):
    def __init__(self, linop=None):
        super().__init__(linop)

    def _prox(self, v, lam):
        return soft_threshold(v, lam)


class norm2(ProxFn):
    def __init__(self, linop=None):
        super().__init__(linop)

    def _prox(self, v, lam):
        return v / (1 + 2 * lam)
