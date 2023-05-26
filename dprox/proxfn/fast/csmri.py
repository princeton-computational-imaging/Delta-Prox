import torch
import torch.fft

from ..sum_square import ext_sum_squares
from dprox.utils import fft2, ifft2


class csmri(ext_sum_squares):
    def __init__(self, linop, mask, y):
        super().__init__(linop)
        self.mask = mask
        self.y = y

    def _prox(self, v, lam, num_psi):
        if len(lam.shape) == 1:
            lam = lam.view(lam.shape[0], 1, 1, 1)
        y = self.unwrap(self.y)
        mask = self.unwrap(self.mask).bool()

        z = fft2(v)
        temp = ((lam * z.clone()) + y) / (1 + lam * num_psi)
        z[mask] = temp[mask]
        z = ifft2(z)

        return z
