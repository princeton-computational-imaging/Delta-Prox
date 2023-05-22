from dprox import *
from dprox.utils import *
from dprox.utils.examples import *

img = sample()
img = to_torch_tensor(img, batch=True).float()
psf = point_spread_function(15, 5)
offset = blurring(img, psf)
offset = mosaicing(offset)

x = Variable()
data_term = sum_squares(mosaic(conv(x, psf)), offset)
reg_term = deep_prior(x, denoiser='ffdnet_color')
prob = Problem(data_term + reg_term,
               lin_solver_kwargs=dict(
                   num_iters=500,
                   verbose=False,
                   lin_solver_type='cg2'
               ),
               absorb=True
               )

out = prob.solve(method='admm', x0=offset, max_iter=24, pbar=True)


imshow(out)
print(psnr(out, img))  # 29.689
