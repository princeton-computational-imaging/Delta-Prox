# %%
import sys
sys.path.append('../')

import torch

from dprox import *
from dprox.utils import *
from dprox import Variable
from dprox.linop.conv import conv_doe

from doe_model import img_psf_conv
from common import (build_doe_model,build_baseline_profile, 
                    circular, load_sample_img, sanity_check, device)

x = Variable()
y = Placeholder()
PSF = Placeholder()
data_term = sum_squares(conv_doe(x, PSF, circular=circular), y)
reg_term = deep_prior(x, denoiser='ffdnet_color')
solver = compile(data_term + reg_term, method='admm')
rgb_collim_model = build_doe_model()
fresnel_phase_c = build_baseline_profile(rgb_collim_model)

sigma = 7.65 / 255
max_iter = 10
rhos, sigmas = log_descent(49, 7.65, max_iter, sigma=max(0.255 / 255, sigma))

def step_fn(gt):
    gt = gt.to(device).float()
    # psf = rgb_collim_model.get_psf()
    psf = rgb_collim_model.get_psf(fresnel_phase_c)
    inp = img_psf_conv(gt, psf, circular=circular)
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
pred = pred.clamp(0, 1)

imshow(gt, inp, pred)
print(psnr(pred, gt))  # 21.28
# %%
