from dprox import *
from dprox.utils import *
from dprox.contrib import *

img = sample()
img = to_torch_tensor(img, batch=True).float()
psf = point_spread_function(15, 5)
offset = blurring(img, psf)
offset = mosaicing(offset)

x = Variable()
data_term = sum_squares(mosaic(conv(x, psf)), offset)
reg_term = deep_prior(x, denoiser='ffdnet_color')
prob = Problem(data_term + reg_term, absorb=True)

out = prob.solve(method='admm', x0=offset, max_iter=24, pbar=True)


imshow(out)
print(psnr(out, img))  # 25.9
