import torch

from dprox.utils.misc import batchify, to_ndarray
from dprox.utils.psf2otf import psf2otf

from .base import LinOp


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

    def forward(self, inputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        input = inputs[0]
        FB = self._FB(input.shape).to(input.device)
        Fx = torch.fft.fftn(input, dim=[-2, -1])
        output = torch.real(torch.fft.ifftn(FB * Fx, dim=[-2, -1])).float()
        return [output]

    def adjoint(self, inputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        input = inputs[0]
        FB = self._FB(input.shape).to(input.device)
        Fx = torch.fft.fftn(input, dim=[-2, -1])
        output = torch.real(torch.fft.ifftn(torch.conj(FB) * Fx, dim=[-2, -1])).float()
        return [output]

    def is_diag(self, freq=False):
        """Is the lin op diagonal (in the frequency domain)?
        """
        return freq and self.input_nodes[0].is_diag(freq)

    def get_diag(self, x, freq=False):
        """Returns the diagonal representation (A^TA)^(1/2).

        Parameters
        ----------
        freq : bool
            Is the diagonal representation in the frequency domain?
        Returns
        -------
        dict of variable to ndarray
            The diagonal operator acting on each variable.
        """
        assert freq
        FB = self._FB(x.shape)
        # var_diags = self.input_nodes[0].get_diag(shape, freq)
        self_diag = torch.abs(torch.conj(FB) * FB)
        # for var in var_diags.keys():
            # var_diags[var] = var_diags[var] * self_diag
        return self_diag.to(self.device)

    def norm_bound(self, input_mags):
        """Gives an upper bound on the magnitudes of the outputs given inputs.

        Parameters
        ----------
        input_mags : list
            List of magnitudes of inputs.

        Returns
        -------
        float
            Magnitude of outputs.
        """
        return np.max(np.abs(self.forward_kernel)) * input_mags[0]

import numpy as np
import torch.nn.functional as F


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

    def __init__(self, arg, psf, circular=False):
        super().__init__([arg])
        self._psf = psf
        self.circular = circular

    def params(self):
        params = super().params()
        params += [self._psf]
        return params
    
    def forward(self, inputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        img = inputs[0]
        psf = self.unwrap(self._psf).to(img.device)
        
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
        
        return [output]

    def adjoint(self, inputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        img = inputs[0]
        psf = self.unwrap(self._psf).to(img.device)
        
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
        
        return [output]

    def is_diag(self, freq=False):
        """Is the lin op diagonal (in the frequency domain)?
        """
        return freq and self.input_nodes[0].is_diag(freq)

    def get_diag(self, x, freq=False):
        """Returns the diagonal representation (A^TA)^(1/2).

        Parameters
        ----------
        freq : bool
            Is the diagonal representation in the frequency domain?
        Returns
        -------
        dict of variable to ndarray
            The diagonal operator acting on each variable.
        """
        assert freq
        psf = self.unwrap(self._psf).to(x.device)
        otf = psf2otf2(psf, x.shape)
        # var_diags = self.input_nodes[0].get_diag(shape, freq)
        self_diag = torch.abs(torch.conj(otf) * otf)
        # for var in var_diags.keys():
            # var_diags[var] = var_diags[var] * self_diag
        return self_diag.to(self.device)

    def norm_bound(self, input_mags):
        """Gives an upper bound on the magnitudes of the outputs given inputs.

        Parameters
        ----------
        input_mags : list
            List of magnitudes of inputs.

        Returns
        -------
        float
            Magnitude of outputs.
        """
        return np.max(np.abs(self.forward_kernel)) * input_mags[0]
