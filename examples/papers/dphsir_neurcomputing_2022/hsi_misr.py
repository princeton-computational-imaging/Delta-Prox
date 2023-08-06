# %%
import cv2
from scipy.io import loadmat
from dprox import *
from dprox.utils import *

from degrades.sr import GaussianDownsample
from degrades.general import HSI2RGB

# ------------------------------------- #
#             Prepare Data              #
# ------------------------------------- #

I = loadmat('Lehavim_0910-1717.mat')['gt'].astype('float32')

sf = 2
down = GaussianDownsample(sf=sf)
y = down(I)

x0 = cv2.resize(y, (y.shape[1] * sf, y.shape[0] * sf), interpolation=cv2.INTER_CUBIC)

srf = HSI2RGB().srf.T  # 31*3

T = mul_color(srf)
z = to_ndarray(T(to_torch_tensor(I, batch=True)), debatch=True)

print(I.shape, y.shape, x0.shape, z.shape)
imshow(I[:, :, 20], y[:, :, 20], x0[:, :, 20], z)

# %%
# ----------------------------------- #
#             Define and Solve        #
# ----------------------------------- #


x = Variable()
data_term = sisr(x, down.kernel, sf, y)
rgb_fidelity = misr(x, z, srf)
reg = deep_prior(x, denoiser='grunet')
problem = Problem(rgb_fidelity + data_term + reg)

rhos, sigmas = log_descent(35, 10, 1)
x_pred = problem.solve(solver='admm',
                       x0=x0,
                       rhos=rhos,
                       weights={reg: sigmas},
                       max_iter=1,
                       pbar=True)

out = to_ndarray(x_pred, debatch=True)
print(mpsnr(out, I))  # 59.11, 59.74
imshow(out[:, :, 20])
# %%
