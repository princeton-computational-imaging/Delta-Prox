from dprox import *
from dprox.utils import *
from dprox.contrib import *

img = sample('face')
psf = point_spread_function(5, 3)
y, x0 = downsampling(img, psf, sf=2)

x = Variable()
data_term = sisr(x, y, kernel=psf, sf=2)
reg_term = deep_prior(x, denoiser='ffdnet_color')
prob = Problem(data_term + reg_term)

max_iter = 24
rhos, sigmas = log_descent(35, 35, max_iter)

out = prob.solve(method='admm', x0=x0, rhos=rhos, lams={reg_term: sigmas}, max_iter=24, pbar=True)

print(psnr(out, img))  # 32.9
imshow(out)