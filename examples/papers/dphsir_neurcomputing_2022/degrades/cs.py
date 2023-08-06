import hdf5storage
import os
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class CASSI(object):
    """ Only work when img size = [512, 512, 31]
    """
    cache = {}
    
    def __init__(self, size=None):
        if size:
            size = tuple(size)
            if size not in CASSI.cache:
                H, W, C = size
                m = np.random.choice([0, 1], size=(H,W), p=[1./2, 1./2])
                ms = [np.roll(m, i, axis=0) for i in range(C)]
                mask = np.stack(ms, axis=2)
                CASSI.cache[size] = mask.astype('float32')
            self.mask = CASSI.cache[size]
        else:
            self.mask = hdf5storage.loadmat(os.path.join(CURRENT_DIR, 'kernels', 'cs_mask_cassi.mat'))['mask']

    def __call__(self, img):
        return np.sum(img * self.mask, axis=2)
