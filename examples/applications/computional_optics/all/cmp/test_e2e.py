import sys
import os
sys.path.append('../')

import torch
import torchlight as tl

from dprox import *
from dprox.utils import *
from dprox import Variable
from dprox.linop.conv import conv_doe

from doe_model import img_psf_conv
from common import (build_doe_model,build_baseline_profile, 
                    circular, load_sample_img, sanity_check, device)
from unet import U_Net


def pad_mod(x, mod):
    h, w = x.shape[-2:]
    h_out = (h // mod + 1) * mod
    w_out = (w // mod + 1) * mod
    out = torch.zeros(*x.shape[:-2], h_out, w_out).type_as(x)
    out[..., :h, :w] = x
    return out.to(x.device), h, w

import time


def main():
    solver = U_Net(3,3).to(device)
    rgb_collim_model = build_doe_model()

    sigma = 7.65 / 255

    def step_fn(gt):
        torch.cuda.synchronize()
        start_time = time.perf_counter()


        gt = gt.to(device).float()
        psf = rgb_collim_model.get_psf()
        inp = img_psf_conv(gt, psf, circular=circular)
        inp = inp + torch.randn(*inp.shape, device=inp.device) * sigma
        
       
        with torch.no_grad():
            out = solver(inp)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        print(elapsed)
        
        return gt, inp, out
    
    method= 'train_deconv_doe'
    
    rgb_collim_model.load_state_dict(torch.load(os.path.join('saved', method, 'best.pth'))['model'])
    solver.load_state_dict(torch.load(os.path.join('saved', method, 'best.pth'))['solver'])    
    
    psf = rgb_collim_model.get_psf()
    # import torchvision
    # torchvision.utils.save_image(psf, 'psf2.png')
    # return 
    tl.metrics.set_data_format('chw')

    # datasets = ['McMaster', 'Kodak24']
    datasets = ['Urban100']

    for dataset in datasets:
        root = 'data/test/' + dataset
        logger = tl.logging.Logger(os.path.join('results',method, dataset), name=dataset)
        tracker = tl.trainer.util.MetricTracker()
        
        timer = tl.utils.Timer()
        timer.tic()
        for idx, name in enumerate(os.listdir(root)):
            gt = load_sample_img(os.path.join(root, name))

            gt, h, w = pad_mod(gt, 16)

            torch.manual_seed(idx)
            torch.cuda.manual_seed(idx)
            gt, inp, pred = step_fn(gt)
            pred = pred.clamp(0, 1)
            
            gt = gt[...,:h,:w]
            pred = pred[...,:h,:w]
            inp = inp[...,:h,:w]
            
            psnr = tl.metrics.psnr(pred, gt)
            ssim = tl.metrics.ssim(pred, gt)
            
            tracker.update('psnr', psnr)
            tracker.update('ssim', ssim)
            
            logger.info('{} PSNR {} SSIM {}'.format(name, psnr, ssim))  
            logger.save_img(name+'_pred.png', pred)
            logger.save_img(name+'_inp.png', inp)
            logger.save_img(name+'_gt.png', gt)

        logger.info('averge results')
        logger.info(tracker.summary())
        print(timer.toc()/len(os.listdir(root)))
        

if __name__ == '__main__':
    main()