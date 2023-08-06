import torch
import torch.nn as nn

from dprox import ProxFn
from network import Denoiser, ResBlock, default_conv


class dgu_prior(ProxFn):
    def __init__(self, linop, denoiser=None):
        super().__init__(linop)
        if denoiser is not None:
            self.denoiser = denoiser()
        else:
            self.denoiser = Denoiser()

    def eval(self, v):
        raise NotImplementedError('deep prior cannot be explictly evaluated')

    def _prox(self, v: torch.Tensor, lam: torch.Tensor = None):
        """ v: [N, C, H, W]
            lam: [1]
        """
        out = self.denoiser(v, self.step)
        return out


class dgu_linop(nn.Module):
    def __init__(self, diag=False):
        super().__init__()
        self.phi_0 = ResBlock(default_conv, 3, 3)
        self.phi_1 = ResBlock(default_conv, 3, 3)
        self.phi_6 = ResBlock(default_conv, 3, 3)
        self.phit_0 = ResBlock(default_conv, 3, 3)
        self.phit_1 = ResBlock(default_conv, 3, 3)
        self.phit_6 = ResBlock(default_conv, 3, 3)

        if diag:
            self.phid_0 = ResBlock(default_conv, 3, 3)
            self.phid_1 = ResBlock(default_conv, 3, 3)
            self.phid_6 = ResBlock(default_conv, 3, 3)

        self.max_step = 5
        self.step = 0

    def forward(self, x, step=None):
        if step is None: step = self.step
        if step == 0:
            return self.phi_0(x)
        elif step == self.max_step + 1:
            return self.phi_6(x)
        else:
            return self.phi_1(x)

    def adjoint(self, x, step=None):
        if step is None: step = self.step
        if step == 0:
            return self.phit_0(x)
        elif step == self.max_step + 1:
            return self.phit_6(x)
        else:
            return self.phit_1(x)

    def diag(self, x, step=None):
        if step is None: step = self.step
        if step == 0:
            return self.phid_0(x)
        elif step == self.max_step + 1:
            return self.phid_6(x)
        else:
            return self.phid_1(x)
