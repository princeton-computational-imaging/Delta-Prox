# %%
from dprox import *
from dprox.utils import *
from dprox.utils import examples

x0, y0, gt, mask = examples.csmri.sample()

x = Variable()
y = Placeholder()
data_term = csmri(x, mask, y)
reg_term = deep_prior(x, denoiser='unet')
prob = Problem(data_term + reg_term)

y.value = y0
max_iter = 24
rhos, sigmas = log_descent(30, 20, max_iter)
prob.solve(
    method='admm',
    device='cuda',
    x0=x0, rhos=rhos, lams={reg_term: sigmas}, max_iter=max_iter, pbar=True
)
out = x.value

print(psnr(out, gt)) # 36.11
imshow(out)

# %%
