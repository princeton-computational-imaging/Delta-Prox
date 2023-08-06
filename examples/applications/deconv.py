from dprox import *
from dprox.utils import *
from dprox.contrib import *

img = sample()
psf = point_spread_function(15, 5)
b = blurring(img, psf)

x = Variable()
data_term = sum_squares(conv(x, psf) - b)
reg_term = deep_prior(x, denoiser='ffdnet_color')
prob = Problem(data_term + reg_term)
prob.solve(method='admm', x0=b)

print(psnr(x.value, img))  # 35
imshow(x.value)