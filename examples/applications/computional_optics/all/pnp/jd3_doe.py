# %%
import sys
sys.path.append('../')

import torch
from common import (build_baseline_profile, build_doe_model, circular, device,
                    load_sample_img, sanity_check)
from doe_model import img_psf_conv

from dprox import *
from dprox import Variable
from dprox.linop.conv import conv_doe
from dprox.utils import *

rgb_collim_model = build_doe_model()
fresnel_phase_c = build_baseline_profile(rgb_collim_model)

x = Variable()
y = Placeholder()
PSF = Placeholder()
data_term = sum_squares(mosaic(conv_doe(x, PSF, circular=circular)), y)
reg_term = deep_prior(x, denoiser='ffdnet_color', trainable=False)
solver = compile(data_term + reg_term, method='admm',
                 linear_solve_config=dict(
                     num_iters=50,
                     verbose=False,
                     lin_solver_type='custom2',
                     tol=1e-6,
                     height_map=rgb_collim_model.height_map,
                 ))

sigma = 7.65 / 255
max_iter = 10
rhos, sigmas = log_descent(49, 7.65, max_iter, sigma=max(0.255 / 255, sigma))


def step_fn(gt):
    gt = gt.to(device).float()
    # psf = rgb_collim_model.get_psf()
    psf = rgb_collim_model.get_psf(fresnel_phase_c)
    inp = img_psf_conv(gt, psf, circular=circular)
    inp = mosaicing(inp)
    inp = inp + torch.randn(*inp.shape, device=inp.device) * sigma
    y.value = inp
    PSF.value = psf
    out = solver.solve(x0=inp,
                       rhos=rhos,
                       lams={reg_term: sigmas},
                       max_iter=max_iter)
    return gt, inp, out


x = load_sample_img()
gt, ob = sanity_check(rgb_collim_model.get_psf())

gt, inp, pred = step_fn(gt)
# pred.mean().backward()
# print(rgb_collim_model.height_map.height_map_sqrt.grad)


pred = pred.clamp(0, 1)

imshow(gt, inp, pred)
print(psnr(pred, gt))  # 20.75
# %%
