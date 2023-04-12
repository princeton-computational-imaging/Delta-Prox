import torch

from ..sum_square import ext_sum_squares


class spi(ext_sum_squares):
    def __init__(self, linop, K, y):
        super().__init__(linop, y)
        self.K = self.to_parameter(K)
        self.x0 = self.to_parameter(y)

    def _prox(self, v, lam):
        assert self.I == torch.ones_like(self.I), \
            'spi ext_sum_square only support I=1'
            
        K = self.K.value * 10
        K1 = self.x0.value * (K ** 2)

        out = spi_inverse(v, K1, K, lam)
        return out


def kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]

    return res.reshape(siz0 + siz1)


def spi_forward(x, K, alpha, q):
    ones = torch.ones(1, 1, K, K).to(x.device)
    theta = alpha * kron(x, ones) / (K**2)
    y = torch.poisson(theta)
    ob = (y >= torch.ones_like(y) * q).float()

    return ob


def spi_inverse(ztilde, K1, K, mu):
    """
    Proximal operator "Prox\_{\frac{1}{\mu} D}" for single photon imaging
    assert alpha == K and q == 1
    """
    z = torch.zeros_like(ztilde)

    K0 = K**2 - K1
    indices_0 = (K1 == 0)

    z[indices_0] = ztilde[indices_0] - (K0 / mu)[indices_0]
    def func(y): return K1 / (torch.exp(y) - 1) - mu * y - K0 + mu * ztilde

    indices_1 = torch.logical_not(indices_0)

    # differentiable binary search
    bmin = 1e-5 * torch.ones_like(ztilde)
    bmax = 1.1 * torch.ones_like(ztilde)

    bave = (bmin + bmax) / 2.0

    for i in range(10):
        tmp = func(bave)
        indices_pos = torch.logical_and(tmp > 0, indices_1)
        indices_neg = torch.logical_and(tmp < 0, indices_1)
        indices_zero = torch.logical_and(tmp == 0, indices_1)
        indices_0 = torch.logical_or(indices_0, indices_zero)
        indices_1 = torch.logical_not(indices_0)

        bmin[indices_pos] = bave[indices_pos]
        bmax[indices_neg] = bave[indices_neg]
        bave[indices_1] = (bmin[indices_1] + bmax[indices_1]) / 2.0

    z[K1 != 0] = bave[K1 != 0]
    return torch.clamp(z, 0.0, 1.0)
