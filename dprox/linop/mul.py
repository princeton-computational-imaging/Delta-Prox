from dprox.utils.misc import to_nn_parameter, to_torch_tensor

from .base import LinOp


class mul_color(LinOp):
    def __init__(self, srf):
        super().__init__()
        # srf: [C, C_2]
        srf = to_torch_tensor(srf).float()
        self.srf = to_nn_parameter(srf)

    def forward(self, x):
        return self.apply(x, self.srf)

    def adjoint(self, x):
        return self.apply(x, self.srf.T)

    def apply(self, x, srf):
        N, C, H, W = x.shape
        x = x.reshape(N, C, H * W)  # N,C,HW
        out = srf.T @ x  # N,C2,HW
        out = out.reshape(N, -1, H, W)
        return out


class mul_elementwise(LinOp):
    def __init__(self, w):
        super().__init__()
        self.w = self.to_parameter(w)

    def forward(self, x):
        return self.w.value * x

    def adjoint(self, x):
        return self.w.value * x

    def is_diagonalizable(self, freq=False):
        return not freq

    def diag(self, freq=False):
        if not freq:
            return self.w.value
        return None
