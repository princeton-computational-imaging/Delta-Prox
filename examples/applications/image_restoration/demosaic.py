from dprox import *
from dprox.utils import *
from dprox.utils.examples import *

img = sample('face')
offset = mosaicing(img)

x = Variable()
data_term = sum_squares(mosaic(x), offset)
reg_term = deep_prior(x, denoiser='ffdnet_color')
prob = Problem(data_term + reg_term, merge=False)

max_iter = 100
rhos, sigmas = log_descent(35, 30, max_iter)
out = prob.solve(method='admm', x0=offset, pbar=True)

print(psnr(out, img))  # 39.027
imshow(out)
