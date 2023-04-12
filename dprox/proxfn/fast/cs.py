import torch

from ..sum_square import ext_sum_squares


class compress_sensing(ext_sum_squares):
    def __init__(self, linop, mask, y):
        super().__init__(linop, y)
        self.y = self.to_parameter(y)
        self.mask = self.to_parameter(mask)

    def _reload(self, shape):
        mask = self.mask.value.float()

        self.phi = torch.sum(mask**2, dim=1, keepdim=True)
        def A(x): return torch.sum(x*mask, dim=1, keepdim=True)
        def At(x): return x*mask

        self.A, self.At = A, At

    def _prox(self, v, lam):
        y, A, At, phi = self.y.value, self.A, self.At, self.phi
        I = self.I

        rhs = At((I*y-A(v))/(phi+I*lam))
        v = (v + rhs)/I
        return v
