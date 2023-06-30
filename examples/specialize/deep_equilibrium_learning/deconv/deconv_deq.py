from dprox import *
from dprox.utils import *

img = sample('face')
psf = point_spread_function(15, 5)
b = blurring(img, psf)

x = Variable()
y = Placeholder()
data_term = sum_squares(conv(x, psf) - y)
reg_term = deep_prior(x, denoiser='ffdnet_color')
solver = compile(data_term + reg_term, method='admm')
solver = DEQSolver(solver)

max_iter = 1
rhos, sigmas = log_descent(30, 30, max_iter)
b = b.cuda()
y.value = b.cuda()
out = solver.solve(x0=b, rhos=rhos, lams={reg_term: sigmas}, max_iter=max_iter)

print(psnr(out, img))  # 31.477
imshow(out)
