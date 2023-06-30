# %%
import torch

from dprox import *
from dprox.utils import *

img = imread_rgb('data/test/Kodak24/kodim01.png') / 255
img = to_torch_tensor(img, batch=True).float()
psf = point_spread_function(15, 5)
print(psf.shape)
b = blurring(img, psf)
b = b+torch.randn_like(b)*7.65/255.

x = Variable()
data_term = sum_squares(conv(x, psf) - b)
reg_term = deep_prior(x, denoiser='ffdnet_color')
prob = Problem(data_term + reg_term)

x.value = b
max_iter = 8
rhos, sigmas = log_descent(49, 7.65, max_iter, sigma=7.65/255)
# rhos, _ = log_descent(1, 0.1, max_iter)
rhos, sigmas = to_torch_tensor(rhos), to_torch_tensor(sigmas)
out = prob.solve(method='admm', x0=b, rhos=rhos, lams={reg_term: sigmas}, max_iter=max_iter)

out = out.clamp(0,1)
imshow(img, b, out)
print(psnr(out, img))  # 38.55

# %%
