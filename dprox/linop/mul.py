from typing import Union

import torch
import torch.nn as nn
import numpy as np

from dprox.utils.misc import to_torch_tensor

from .base import LinOp
from .placeholder import Placeholder


class mul_color(LinOp):
    def __init__(
        self,
        arg: LinOp,
        srf: Union[Placeholder, torch.Tensor, np.array],
    ):
        super().__init__([arg])
        # srf: [C, C_2]
        self._srf = srf

        if isinstance(srf, Placeholder):
            def on_change(val):
                self.srf = nn.parameter.Parameter(val)
            self._srf.change(on_change)
        else:
            self.srf = nn.parameter.Parameter(to_torch_tensor(srf, batch=True))

    def forward(self, x, **kwargs):
        return self.apply(x, self.srf)

    def adjoint(self, x, **kwargs):
        return self.apply(x, self.srf.T)

    def apply(self, x, srf):
        N, C, H, W = x.shape
        x = x.reshape(N, C, H * W)  # N,C,HW
        out = srf.T @ x  # N,C2,HW
        out = out.reshape(N, -1, H, W)
        return out


class mul_elementwise(LinOp):
    def __init__(
        self,
        arg: LinOp,
        w: Union[Placeholder, torch.Tensor, np.array],
    ):
        super().__init__([arg])
        self._w = w

        if isinstance(w, Placeholder):
            def on_change(val):
                self.w = nn.parameter.Parameter(val)
            self._w.change(on_change)
        else:
            self.w = nn.parameter.Parameter(to_torch_tensor(w, batch=True))

    def forward(self, x, **kwargs):
        w = self.w.to(x.device)
        return w * x

    def adjoint(self, x, **kwargs):
        return self.forward(x)

    def is_diag(self, freq=False):
        return not freq

    def get_diag(self, x, freq=False):
        if not freq:
            return self.w.to(x.device)
        return None
