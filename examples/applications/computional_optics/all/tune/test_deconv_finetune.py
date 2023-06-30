import sys
import os
from pathlib import Path
sys.path.append('../')

import torch
import torch.nn as nn
import torchlight as tl
import torchlight.nn as tlnn
import torchvision.utils
from common import (build_baseline_profile, build_doe_model, circular, device,
                    load_sample_img, plot, sanity_check, subplot)
from doe_model import img_psf_conv

from dprox import *
from dprox import Variable
from dprox.linop.conv import conv_doe
from dprox.utils import *


from train_jd3 import mosaicing

@torch.no_grad()
def main():
    rgb_collim_model = build_doe_model()
    
    # -------------------- Define Model --------------------- #
    x = Variable()
    y = Placeholder()
    PSF = Placeholder()
    data_term = sum_squares(conv_doe(x, PSF, circular=circular), y)
    # data_term = sum_squares(mosaic(conv_doe(x, PSF, circular=circular)), y)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    solver = compile(data_term + reg_term, method='admm',
                     linear_solve_config=dict(
                        num_iters=50,
                        verbose=False,
                        lin_solver_type='custom2',
                        tol=1e-5,
                        height_map=rgb_collim_model.height_map,
                 ))

    # ---------------- Setup Hyperparameter ----------------- #
    max_iter = 10
    sigma = 7.65 / 255.
    rhos, sigmas = log_descent(49, 7.65, max_iter, sigma=max(0.255 / 255, sigma))
    rhos = torch.tensor(rhos, device=device).float()
    sigmas = torch.tensor(sigmas, device=device).float()
    rgb_collim_model.rhos = nn.parameter.Parameter(rhos)
    rgb_collim_model.sigmas = nn.parameter.Parameter(sigmas)

    # ---------------- Forward Model ------------------------ #
    def step_fn(gt):
        gt = gt.to(device).float()
        psf = rgb_collim_model.get_psf()
        inp = img_psf_conv(gt, psf, circular=circular)
        inp = inp + torch.randn(*inp.shape, device=inp.device) * sigma
        y.value = inp
        PSF.value = psf
        out = solver.solve(x0=inp,
                           rhos=rgb_collim_model.rhos,
                           lams={reg_term: rgb_collim_model.sigmas},
                           max_iter=max_iter)
        return gt, inp, out

    # ----------------- Sanity Check ------------------------ #
    gt, ob = sanity_check(rgb_collim_model.get_psf())
    gt, inp, out = step_fn(gt)
    print(gt.shape, ob.shape, out.shape)
    print(tl.metrics.psnr(out, gt))  # 38.55
    imshow(gt, ob, out)
    
    print('Model size (M)', tlnn.benchmark.model_size(rgb_collim_model))
    print('Trainable model size (M)', tlnn.benchmark.trainable_model_size(rgb_collim_model))

    # --------------- Load and Run ------------------ #
    
    method = 'train_deconv_finetune'
    savedir = Path('saved') / method

    ckpt = torch.load(savedir / 'best.pth')
    rgb_collim_model.load_state_dict(ckpt['model'])
    solver.load_state_dict(ckpt['solver'], strict=False)

    
    tl.metrics.set_data_format('chw')

    datasets = ['McMaster', 'Kodak24']
    datasets = ['Urban100']
    

    for dataset in datasets:
        root = 'data/test/' + dataset
        logger = tl.logging.Logger(os.path.join('results/', method, dataset), name=dataset)
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
            logger.save_img(name+'_inp.png', inp)
            logger.save_img(name+'_gt.png', gt)

        logger.info('averge results')
        logger.info(tracker.summary())
        print(timer.toc()/len(os.listdir(root)))

    

    # # --------------- Save Results ------------------ #
    # torchvision.utils.save_image(inp, imgdir / f'inp.png')
    # torchvision.utils.save_image(pred, imgdir / f'pred.png')
    # torchvision.utils.save_image(gt, imgdir / f'gt.png')

    # height_map = rgb_collim_model.height_map.height_map_sqrt.detach()**2
    # plot(height_map, imgdir / f'height_map.png')
    # psf = rgb_collim_model.get_psf()
    # subplot(psf, imgdir / f'psf.png')
    # import matplotlib.pyplot as plt
    # img = psf[0].permute(1,2,0).detach().cpu().numpy()
    # # img = np.log(img)
    # plt.figure()
    # plt.imshow(img)
    # plt.colorbar()
    # plt.savefig(imgdir/'psf_rgb.png')
    # phase = torch.angle(rgb_collim_model.height_map.get_phase_profile())
    # subplot(phase, imgdir / f'phase.png')


main()
