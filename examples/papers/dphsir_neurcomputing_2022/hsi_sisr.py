# %%
import cv2
import torch
from scipy.io import loadmat

from dprox import *
from dprox.utils import *

from degrades import GaussianDownsample

# ------------------------------------- #
#             Prepare Data              #
# ------------------------------------- #

I = loadmat('Lehavim_0910-1717.mat')['gt'].astype('float32')

sf = 2
down = GaussianDownsample(sf=sf)
y = down(I)

x0 = cv2.resize(y, (y.shape[1]*sf, y.shape[0]*sf), interpolation=cv2.INTER_CUBIC)

print(I.shape, y.shape, x0.shape)
imshow(I[:, :, 20], y[:, :, 20], x0[:, :, 20])


#%%
# ----------------------------------- #
#             Define and Solve        #
# ----------------------------------- #

x = Variable()
data_term = sisr(x, down.kernel, sf, y)
deep_reg = deep_prior(x, denoiser='grunet')

solver = ADMM([data_term, deep_reg], partition=True)
solver.to(torch.device('cuda'))

rhos, sigmas = log_descent(35, 10, 24)
x_pred = solver.solve(x0, 
                      rhos=rhos,
                      weights={deep_reg: sigmas},
                      max_iter=24,
                      pbar=True)

out = to_ndarray(x_pred, debatch=True)
print(mpsnr(out, I))  # 47.9153/47.5494
imshow(out[:, :, 20])
# %%
