import abc
import torch
import torch.nn as nn

class Denoiser(nn.Module):
    def denoise(self, input: torch.Tensor, sigma: torch.Tensor):
        """ x: [NCHW] , sigma: a single number tensor"""
        sigma = sigma.view(-1, 1, 1, 1)
        output = self._denoise(input, sigma)
        return output

    @abc.abstractmethod
    def _denoise(self, x, sigma):
        raise NotImplementedError


class Denoiser2D(Denoiser):
    def denoise(self, input: torch.Tensor, sigma: torch.Tensor):
        """ x: [NCHW] , sigma: a single number tensor"""
        sigma = sigma.view(-1, 1, 1, 1)
        outs = []
        for band in input.split(1, dim=1):
            band = self._denoise(band, sigma)
            outs.append(band)
        return torch.cat(outs, dim=1)
