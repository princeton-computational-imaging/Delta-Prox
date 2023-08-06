# %%
from scipy.io import loadmat

from dprox import *
from dprox.utils import *

from degrades.inpaint import FastHyStripe

# ------------------------------------- #
#             Prepare Data              #
# ------------------------------------- #

I = loadmat('Lehavim_0910-1717.mat')['gt']

degrade = FastHyStripe()
b, mask = degrade(I)
mask = mask.astype('float')

imshow(I[:,:,20], b[:,:,20])

#%%
# ----------------------------------- #
#             Define and Solve        #
# ----------------------------------- #

S = MulElementwise(mask)

x = Variable()
data_term = sum_squares(S, b)
deep_reg = deep_prior(x, denoiser='grunet')
problem = Problem(data_term+deep_reg)

rhos, sigmas = log_descent(5, 4, iter=24, lam=0.6)
x_pred = problem.solve(solver='admm', 
                       x0=b, 
                       weights={deep_reg: sigmas},
                       rhos=rhos,
                       max_iter=24,
                       pbar=True)

out = to_ndarray(x_pred, debatch=True)

print(mpsnr(out, I)) # 74.92/74.88
imshow(out[:,:,20])

# %%
