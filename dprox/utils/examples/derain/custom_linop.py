# Learnable Linear Operator from
# https://github.com/MC-E/Deep-Generalized-Unfolding-Networks-for-Image-Restoration/blob/main/Deraining/DGUNet.py
# Deep Generalized Unfolding Networks for Image Restoration

import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(), res_scale=1
    ):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            if i == 0:
                m.append(conv(n_feats, 64, kernel_size, bias=bias))
            else:
                m.append(conv(64, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class LearnableDegOp(nn.Module):
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

