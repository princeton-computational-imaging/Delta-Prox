# %%
import torch
from scipy.io import loadmat

from dprox import *
from dprox.utils import *

from degrades.blur import GaussianBlur

# ------------------------------------- #
#             Prepare Data              #
# ------------------------------------- #


I = loadmat('Lehavim_0910-1717.mat')['gt']

blur = GaussianBlur()
b = blur(I)

print(I.shape, b.shape)
imshow(I[:, :, 20], b[:, :, 20])

# %%
# ----------------------------------- #
#             Define and Solve        #
# ----------------------------------- #

x = Variable()
data_term = sum_squares(conv(x, blur.kernel), b)
reg_term = deep_prior(x, denoiser='grunet')

device = torch.device('cuda')
solver = compile(data_term + reg_term)

iter_num = 24
rhos, sigmas = log_descent(35, 10, iter_num)
x_pred = solver.solve(to_torch_tensor(b, batch=True).to(device),
                      rhos=1e-10,
                      weights={reg_term: sigmas},  # 54.97 if reg_term: 0.23
                      max_iter=iter_num,
                      eps=0,
                      pbar=True)

out = to_ndarray(x_pred, debatch=True)
print(mpsnr(out, I))  # 54.22/55.10, 78.31 if rho = 1e-10
imshow(out[:, :, 20])

# two deep prior: 53.00

# %%
