
import os
import abc

import hdf5storage
import numpy as np

from .utils import imresize_np
from .blur import AbstractBlur, GaussianBlur, UniformBlur

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class AbstractDownsample(abc.ABC):
    def __init__(self, sf, kernel):
        self.sf = sf
        self.kernel = kernel


class ClassicalDownsample(AbstractDownsample):
    def __init__(self, sf, blur: AbstractBlur):
        super().__init__(sf, blur.kernel)
        self.blur = blur

    def __call__(self, img):
        """ input: [w,h,c]
            data range: both (0,255), (0,1) are ok
        """
        img = self.blur(img)
        img = img[0::self.sf, 0::self.sf, ...]
        return img


class GaussianDownsample(ClassicalDownsample):
    def __init__(self, sf, ksize=8, sigma=3):
        blur = GaussianBlur(ksize, sigma)
        super().__init__(sf, blur)


class UniformDownsample(ClassicalDownsample):
    def __init__(self, sf):
        blur = UniformBlur(sf)
        super().__init__(sf, blur)


class BiCubicDownsample(AbstractDownsample):
    kernel_path = os.path.join(CURRENT_DIR, 'kernels', 'kernels_bicubicx234.mat')
    valid_sfs = [2, 3, 4]

    def __init__(self, sf):
        if sf not in self.valid_sfs:
            raise ValueError(f'Invalid scale factor, choose from {self.valid_sfs}')
        self.sf = sf
        self.kernels = hdf5storage.loadmat(self.kernel_path)['kernels']
        self.kernel = self.kernels[0, sf-2].astype(np.float64)

    def __call__(self, img):
        """ input: [w,h,c]
            data range: both (0,255), (0,1) are ok
        """
        img_L = imresize_np(img, 1/self.sf)
        return img_L
