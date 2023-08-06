# %%
import numpy as np
from scipy.io import loadmat
import torch

from dprox import *
from dprox.utils import *

from degrades.cs import CASSI

# ------------------------------------- #
#             Prepare Data              #
# ------------------------------------- #

I = loadmat('Lehavim_0910-1717.mat')['gt']

down = CASSI()
mask = down.mask.astype('float32')
y = down(I).astype('float32')

x0 = np.expand_dims(y, axis=-1) * mask

print(I.shape, y.shape, x0.shape)
imshow(I[:, :, 20], y, x0[:, :, 20])

# %
# ----------------------------------- #
#             Define and Solve        #
# ----------------------------------- #

x = Variable()
data_term = compress_sensing(x, mask, y)
reg_term = deep_prior(x, denoiser='grunet')

solver = compile(data_term+reg_term)
solver.to(torch.device('cuda'))

iter_num = 24
rhos, sigmas = log_descent(50, 45, iter_num)
x_pred = solver.solve(x0,
                      rhos=rhos,
                      weights={reg_term: sigmas},
                      max_iter=iter_num,
                      pbar=True)

out = to_ndarray(x_pred, debatch=True)

print(mpsnr(out, I))  # mpsnr: 39.18
imshow(out[:, :, 20])
# %%
