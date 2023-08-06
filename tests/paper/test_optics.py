import os

import torch
import torch.nn as nn
import torchlight as tl
import torchlight.nn as tlnn

from dprox import *
from dprox import Variable
from dprox.linop.conv import conv_doe
from dprox.utils import *
from dprox.contrib.optic.utils import load_sample_img
from dprox.contrib.optic.doe_model import img_psf_conv, build_doe_model, DOEModelConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def run_optics(dataset):
    config = DOEModelConfig()
    rgb_collim_model = build_doe_model().to(device)
    circular = config.circular

    # -------------------- Define Model --------------------- #
    x = Variable()
    y = Placeholder()
    PSF = Placeholder()
    data_term = sum_squares(conv_doe(x, PSF, circular=circular), y)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    solver = compile(data_term + reg_term, method='admm')
    solver.eval()

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

        timer = tl.utils.Timer()
        timer.tic()

        out = solver.solve(x0=inp,
                           rhos=rgb_collim_model.rhos,
                           lams={reg_term: rgb_collim_model.sigmas.sqrt()},
                           max_iter=max_iter)

        return gt, inp, out

    print('Model size (M)', tlnn.benchmark.model_size(rgb_collim_model))
    print('Trainable model size (M)', tlnn.benchmark.trainable_model_size(rgb_collim_model))

    # --------------- Load and Run ------------------ #
    ckpt = hf.load_checkpoint('computational_optics/joint_dprox.pth')
    rgb_collim_model.load_state_dict(ckpt['model'])

    tl.metrics.set_data_format('chw')

    root = dataset
    root = hf.download_dataset(root)
    tracker = tl.trainer.util.MetricTracker()

    timer = tl.utils.Timer()
    timer.tic()
    for idx, name in enumerate(list_image_files(root)):
        gt = load_sample_img(os.path.join(root, name))

        torch.manual_seed(idx)
        torch.cuda.manual_seed(idx)
        gt, inp, pred = step_fn(gt)
        pred = pred.clamp(0, 1)

        psnr = tl.metrics.psnr(pred, gt)
        ssim = tl.metrics.ssim(pred, gt)

        tracker.update('psnr', psnr)
        tracker.update('ssim', ssim)

        print('{} PSNR {} SSIM {}'.format(name, psnr, ssim))

    print('averge results')
    print(tracker.summary())
    print(timer.toc() / len(list_image_files(root)))
    return tracker['psnr'], tracker['ssim']


def test_ubran100():
    psnr, ssim = run_optics('Urban100')
    assert abs(psnr - 30.83) < 0.01
    assert abs(ssim - 0.944) < 0.001


def test_cbsd68():
    psnr, ssim = run_optics('CBSD68')
    assert abs(psnr - 32.01) < 0.01
    assert abs(ssim - 0.942) < 0.001
