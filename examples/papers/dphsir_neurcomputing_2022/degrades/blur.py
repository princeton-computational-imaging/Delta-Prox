import numpy as np
from scipy import ndimage

from .utils import fspecial_gaussian


class AbstractBlur:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, img):
        # img_L = np.fft.ifftn(np.fft.fftn(img) * np.fft.fftn(np.expand_dims(self.k, axis=2), img.shape)).real
        img_L = ndimage.filters.convolve(img, np.expand_dims(self.kernel, axis=2), mode='wrap')
        return img_L


class GaussianBlur(AbstractBlur):
    def __init__(self, ksize=8, sigma=3):
        k = fspecial_gaussian(ksize, sigma)
        super().__init__(k)


class UniformBlur(AbstractBlur):
    def __init__(self, ksize):
        k = np.ones((ksize, ksize)) / (ksize*ksize)
        super().__init__(k)
