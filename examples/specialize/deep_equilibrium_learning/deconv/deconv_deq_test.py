import argparse

import torch

from dprox import *
from dprox.algo.tune import *
from dprox.utils import *
from dprox.utils import examples

img = sample('face')
psf = point_spread_function(15, 5)
b = blurring(img, psf)

x = Variable()
y = Placeholder()
data_term = sum_squares(conv(x, psf) - y)
reg_term = deep_prior(x, denoiser='ffdnet_color', trainable=True)
solver = compile(data_term + reg_term, method='admm')
solver = DEQSolver(solver)
solver.load_state_dict(torch.load('saved_deconv/last.pth')['solver'])

y.value = b.cuda()
max_iter = 100
rhos, sigmas = log_descent(30, 30, max_iter)
b = b.cuda()
out = solver.solve(x0=b, rhos=rhos, lams={reg_term: sigmas}, max_iter=max_iter)

print(psnr(out, img))  # 31.677
imshow(out)