from dprox import *
from dprox.utils import *
from dprox.contrib import *

img = sample('face')
offset = mosaicing(img)

x = Variable()
data_term = sum_squares(mosaic(x), offset)
reg_term = deep_prior(x, denoiser='ffdnet_color')
prob = Problem(data_term + reg_term, merge=False)

max_iter = 30
rhos, sigmas = log_descent(35, 5, max_iter, sqrt=True)
out = prob.solve(method='admm', x0=offset, pbar=True, 
                 rhos=rhos, lams={reg_term: sigmas})

print(psnr(out, img))  # 39
imshow(out)