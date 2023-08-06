# Demosaic Ref
# - https://sporco.readthedocs.io/en/latest/examples/ppp/ppp_admm_dmsc.html
# - https://github.com/cszn/DPIR/blob/master/main_dpir_demosaick.py

import cv2
import scipy.misc

from dprox import *
from dprox.utils import *

from utils import dm_matlab, mosaic_CFA_Bayer, tensor2uint, uint2tensor4

#  Prepare GT and Input
# I = imread_rgb('/media/exthdd/laizeqiang/lzq/projects/hyper-pnp/OpenProx/examples/experiments/rgb/data/Kodak24/kodim02.png')
I = scipy.misc.face()
CFA, CFA4, b, mask = mosaic_CFA_Bayer(I)

x0 = dm_matlab(uint2tensor4(CFA4))
x0 = tensor2uint(x0)
x0 = cv2.cvtColor(CFA, cv2.COLOR_BAYER_BG2RGB_EA)  # essential for drunet, wo 14, w 41.72
# x0 = b

imshow(b)
I = I.astype('float32') / 255
x0 = x0.astype('float32') / 255
b = b.astype('float32') / 255

print(b.shape, mask.shape)

#  Define and Solve the Problem
x = Variable()

data_term = sum_squares(mosaic(x), b)
deep_reg = deep_prior(x, denoiser='ffdnet_color', x8=True)
problem = Problem(data_term + deep_reg)


hi = 32
low = 2
iter_num = 40
rhos, sigmas = log_descent(hi, low, iter_num)
x_pred = problem.solve(solver='hqs',
                       x0=x0,
                       weights={deep_reg: sigmas},
                       rhos=rhos,
                       max_iter=iter_num,
                       pbar=False)
out = to_ndarray(x_pred, debatch=True).clip(0, 1)
out[mask] = b[mask]
psnr_ = psnr(out, I)


# best: 32 2 44.766043261756685
print(psnr_)
imshow(out)
