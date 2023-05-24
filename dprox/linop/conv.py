from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from dprox.utils.misc import batchify, to_ndarray, to_torch_tensor
from dprox.utils.psf2otf import psf2otf

from .base import LinOp
from .placeholder import Placeholder


class conv(LinOp):
    """Circular convolution of the input with a kernel.
    """

    def __init__(self, arg, kernel):
        self.kernel = to_ndarray(kernel)
        self.cache = {}
        super(conv, self).__init__([arg])

    def _FB(self, shape):
        if shape not in self.cache:
            _, C, H, W = shape
            FB = psf2otf(self.kernel, [H, W, C])
            FB = batchify(torch.from_numpy(FB))
            self.cache[shape] = FB
        return self.cache[shape]

    def forward(self, input, **kwargs):
        FB = self._FB(input.shape).to(input.device)
        Fx = torch.fft.fftn(input, dim=[-2, -1])
        output = torch.real(torch.fft.ifftn(FB * Fx, dim=[-2, -1])).float()
        return output

    def adjoint(self, input, **kwargs):
        FB = self._FB(input.shape).to(input.device)
        Fx = torch.fft.fftn(input, dim=[-2, -1])
        output = torch.real(torch.fft.ifftn(torch.conj(FB) * Fx, dim=[-2, -1])).float()
        return output

    def is_diag(self, freq=False):
        return freq and self.input_nodes[0].is_diag(freq)

    def get_diag(self, x, freq=False):
        assert freq
        FB = self._FB(x.shape)
        # var_diags = self.input_nodes[0].get_diag(shape, freq)
        self_diag = torch.abs(torch.conj(FB) * FB)
        # for var in var_diags.keys():
        # var_diags[var] = var_diags[var] * self_diag
        return self_diag.to(self.device)

    def norm_bound(self, input_mags):
        return np.max(np.abs(self.forward_kernel)) * input_mags[0]


def psf2otf2(psf, output_size):
    _, _, fh, fw = psf.shape

    # pad out to output_size with zeros
    if output_size[2] != fh:
        pad = (output_size[2] - fh) / 2

        if (output_size[2] - fh) % 2 != 0:
            pad_top = pad_left = int(np.ceil(pad))
            pad_bottom = pad_right = int(np.floor(pad))
        else:
            pad_top = pad_left = int(pad) + 1
            pad_bottom = pad_right = int(pad) - 1

        padded = F.pad(input=psf, pad=[pad_left, pad_right, pad_top, pad_bottom], mode="constant")
    else:
        padded = psf

    # circularly shift so center pixel is at 0,0
    padded = torch.fft.ifftshift(padded)
    otf = torch.fft.fft2(padded)
    return otf


class conv_doe(LinOp):
    """Circular convolution of the input with a kernel.
    """

    def __init__(
        self,
        arg: LinOp,
        psf: Union[Placeholder, torch.Tensor, np.array],
        circular: bool = False
    ):
        super().__init__([arg])
        self._psf = psf
        self.circular = circular

        if isinstance(psf, Placeholder):
            def on_change(val):
                self.psf = nn.parameter.Parameter(val)
            self._psf.change(on_change)
        else:
            self.psf = nn.parameter.Parameter(to_torch_tensor(psf, batch=True))

    def forward(self, img, **kwargs):
        psf = self.psf.to(img.device)

        if not self.circular:
            # linearized conv
            target_side_length = 2 * img.shape[2]
            height_pad = (target_side_length - img.shape[2]) / 2
            width_pad = (target_side_length - img.shape[3]) / 2
            pad_top, pad_bottom = int(np.ceil(height_pad)), int(np.floor(height_pad))
            pad_left, pad_right = int(np.ceil(width_pad)), int(np.floor(width_pad))
            img = F.pad(input=img, pad=[pad_left, pad_right, pad_top, pad_bottom], mode="constant")

        otf = psf2otf2(psf, img.shape)
        Fx = torch.fft.fftn(img, dim=[-2, -1])
        output = torch.real(torch.fft.ifftn(otf * Fx, dim=[-2, -1])).float()

        if not self.circular:
            output = output[:, :, pad_top:-pad_bottom, pad_left:-pad_right]

        return output

    def adjoint(self, img, **kwargs):
        psf = self.unwrap(self.psf).to(img.device)

        if not self.circular:
            # linearized conv
            target_side_length = 2 * img.shape[2]
            height_pad = (target_side_length - img.shape[2]) / 2
            width_pad = (target_side_length - img.shape[3]) / 2
            pad_top, pad_bottom = int(np.ceil(height_pad)), int(np.floor(height_pad))
            pad_left, pad_right = int(np.ceil(width_pad)), int(np.floor(width_pad))
            img = F.pad(input=img, pad=[pad_left, pad_right, pad_top, pad_bottom], mode="constant")

        otf = psf2otf2(psf, img.shape)
        Fx = torch.fft.fftn(img, dim=[-2, -1])
        output = torch.real(torch.fft.ifftn(torch.conj(otf) * Fx, dim=[-2, -1])).float()

        if not self.circular:
            output = output[:, :, pad_top:-pad_bottom, pad_left:-pad_right]

        return output

    def is_diag(self, freq=False):
        return freq and self.input_nodes[0].is_diag(freq)

    def get_diag(self, x, freq=False):
        assert freq
        psf = self.unwrap(self.psf).to(x.device)
        otf = psf2otf2(psf, x.shape)
        # var_diags = self.input_nodes[0].get_diag(shape, freq)
        self_diag = torch.abs(torch.conj(otf) * otf)
        # for var in var_diags.keys():
        # var_diags[var] = var_diags[var] * self_diag
        return self_diag.to(self.device)

    def norm_bound(self, input_mags):
        return np.max(np.abs(self.forward_kernel)) * input_mags[0]
