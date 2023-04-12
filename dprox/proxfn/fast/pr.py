import torch
from .. import ProxFn
from ..sum_square import ext_sum_squares

from dprox.utils.misc import to_torch_tensor


class phase_ret(ProxFn):
    def __init__(self, linop):
        super().__init__(linop)

    def prox(self, v, lam):
        # Az = cdp_forward(z, mask)   # Az
        # y_hat = torch.abs(Az)     # |Az|
        # meas_err = y_hat - y0
        # gradient_forward = torch.stack((meas_err/y_hat*Az[...,0], meas_err/y_hat*Az[...,1]), -1)
        # gradient = cdp_backward(gradient_forward, mask)
        # z = z - _tau * (gradient + _mu * (z - (x + u)))
        pass


def cdp_forward(data, mask):
    """
    Compute the forward model of cdp.

    Args:
        data (torch.Tensor): Image_data (batch_size*1*hight*weight). complex
        mask (torch.Tensor): mask (batch_size*sampling_rate*hight*weight).complex

    Returns:
        forward_data (torch.Tensor): the complex field of forward data (batch_size*sampling_rate*hight*weight) complex
    """
    sampling_rate = mask.shape[1]
    x = data.repeat(1, sampling_rate, 1, 1)
    masked_data = x * mask
    forward_data = torch.fft.fft2(masked_data, norm='ortho')
    return forward_data


def cdp_backward(data, mask):
    """
    Compute the backward model of cdp (the inverse operator of forward model).

    Args:
        data (torch.Tensor): Field_data (batch_size*sampling_rate*hight*weight).
        mask (torch.Tensor): mask (batch_size*sampling_rate*hight*weight)

    Returns:
        backward_data (torch.Tensor): the complex field of backward data (batch_size*1*hight*weight)
    """
    Ifft_data = torch.fft.ifft2(data, norm='ortho')
    backward_data = Ifft_data * torch.conj(mask)
    return backward_data.mean(1, keepdim=True)
