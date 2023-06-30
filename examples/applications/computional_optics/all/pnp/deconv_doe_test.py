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
    timer = tl.utils.Timer()
    timer.tic()
    out = solver.solve(x0=inp,
                        rhos=rhos,
                        lams={reg_term: sigmas},
                        max_iter=max_iter)
    print(timer.toc())
    return gt, inp, out
    
    
import os
import torchlight as tl
tl.metrics.set_data_format('chw')

# datasets = ['McMaster', 'Kodak24']
datasets = ['Urban100']

for dataset in datasets:
    root = 'data/test/' + dataset
    logger = tl.logging.Logger('results/deconv_doe/' + dataset, name=dataset)
    tracker = tl.trainer.util.MetricTracker()

    timer = tl.utils.Timer()
    timer.tic()
    for idx, name in enumerate(os.listdir(root)):
        gt = load_sample_img(os.path.join(root, name))

        torch.manual_seed(idx)
        torch.cuda.manual_seed(idx)
        gt, inp, pred = step_fn(gt)
        pred = pred.clamp(0, 1)

        psnr = tl.metrics.psnr(pred, gt)
        ssim = tl.metrics.ssim(pred, gt)
        
        tracker.update('psnr', psnr)
        tracker.update('ssim', ssim)
        
        logger.info('{} PSNR {} SSIM {}'.format(name, psnr, ssim))  
        logger.save_img(name, pred)

    logger.info('averge results')
    logger.info(tracker.summary())
    print(timer.toc()/len(os.listdir(root)))
