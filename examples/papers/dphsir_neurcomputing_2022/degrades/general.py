import os

import cv2
import numpy as np
from scipy.io import loadmat

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class AffineTransform(object):
    def __call__(self, x):
        srcTri = np.array([[0, 0], [x.shape[1] - 1, 0], [0, x.shape[0] - 1]]).astype(np.float32)
        dstTri = np.array([[0, x.shape[1]*0.05], [x.shape[1]*0.99, x.shape[0]*0], [x.shape[1]*0.05, x.shape[0]*0.99]]).astype(np.float32)
        warp_mat = cv2.getAffineTransform(srcTri, dstTri)
        warp_dst = cv2.warpAffine(x, warp_mat, (x.shape[1], x.shape[0]))
        return warp_dst


class PerspectiveTransform(object):
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, img):
        rows, cols, _ = img.shape
        pts1 = np.float32([[0, 0], [rows, 0], [self.shift, cols], [rows-self.shift, cols]])
        pts2 = np.float32([[0, 0], [rows, 0], [0, cols], [rows, cols]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (rows, cols))
        return dst


class HSI2RGB(object):
    def __init__(self, srf=None):
        if srf is None:
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
            self.srf = loadmat(os.path.join(CURRENT_DIR, 'kernels', 'misr_spe_p.mat'))['P']  # (3,31)
        else:
            self.srf = srf

    def __call__(self, img):
        return img @ self.srf.transpose()
