import torch
import numpy as np

from ..sum_square import ext_sum_squares
from dprox.utils.misc import to_nn_parameter, to_torch_tensor


class misr(ext_sum_squares):
    def __init__(self, linop, b, srf, eps=1e-7):
        super().__init__(linop, b, eps)
        self.offset = self.to_parameter(b)
        self.srf = srf

    def _reload(self, shape):
        b = self.offset.value
        srf = self.srf

        srf = to_torch_tensor(srf).float()  # C*3
        b = to_torch_tensor(b, batch=True).float().cpu()
        N, C, H, W = b.shape
        z = b.reshape(N, C, H * W)  # N,3,H*W

        # torch.backends.cuda.matmul.allow_tf32 = False
        T2 = srf @ srf.T    # C,3 @ 3,C = C,C, must compute on gpu or disable tf32
        Ttz = srf @ z   # C,3 @ 3,H*W = N,C,H*W

        self.eye = to_nn_parameter(torch.eye(T2.shape[0]).float()).to(self.device)
        self.T2 = to_nn_parameter(T2).to(self.device)
        self.Ttz = to_nn_parameter(Ttz).to(self.device)

    def _prox(self, v, lam):
        N, C, H, W = v.shape
        eye, T2, Ttz = self.eye, self.T2, self.Ttz

        v = v.reshape(N, C, H * W)
        lam = lam.reshape(N, 1, 1)
        x = torch.inverse(T2 + self.I * lam * eye).matmul(Ttz + lam * v)
        x = x.reshape(N, C, H, W)

        return x


class sisr(ext_sum_squares):
    def __init__(self, linop, y, kernel, sf):
        super().__init__(linop, y)
        self.sf = sf
        self.y = y
        self.k = kernel

    def reload(self):
        k = self.unwrap(self.k)
        y = self.unwrap(self.y)

        h, w = y.shape[-2:]
        self.STy = upsample(y, sf=self.sf)
        self.FB = p2o(k, (h * self.sf, w * self.sf))
        self.FBC = torch.conj(self.FB)
        self.F2B = torch.pow(torch.abs(self.FB), 2)
        self.FBFy = self.FBC * torch.fft.fftn(self.STy, dim=(-2, -1))

    def _prox(self, v, lam, I):
        self.reload()
        FB, FBC, F2B, FBFy = self.FB, self.FBC, self.F2B, self.FBFy
        sf = self.sf

        # lam = lam / 2

        FR = FBFy + torch.fft.fftn(lam * v, dim=(-2, -1))
        x1 = FB.mul(FR)
        FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
        invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
        invWBR = FBR.div(invW + I * lam)
        FCBinvWBR = FBC * invWBR.repeat(1, 1, sf, sf)
        FX = (FR - FCBinvWBR) / (I * lam + 1e-9)
        Xest = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        return Xest


def single2tensor4(img):
    """ convert single ndarray (H,W,C) to 4-D torch tensor (B,C,H,W)"""
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)


def splits(a, sf):
    '''split a into sfxsf distinct blocks
    Args:
        a: NxCxWxH
        sf: split factor
    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 2)
    otf = torch.fft.fftn(otf, dim=(-2, -1))
    return otf


def upsample(x, sf=3):
    '''s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, NxCxWxH
    '''
    N, C, H, W = x.shape
    st = 0
    z = torch.zeros((N, C, H * sf, W * sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z
