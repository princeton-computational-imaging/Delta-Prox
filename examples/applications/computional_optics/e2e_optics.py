import os
from pathlib import Path

import imageio
import fire
import torch
import torch.nn as nn
import torchlight as tl
import torchlight.nn as tlnn

from dprox import *
from dprox import Variable
from dprox.linop.conv import conv_doe
from dprox.utils import *
from dprox.utils.examples.optic.common import (
    build_doe_model, normalize_psf, load_sample_img, DOEModelConfig
)
from dprox.utils.examples.optic.doe_model import img_psf_conv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model():
    config = DOEModelConfig()
    rgb_collim_model = build_doe_model(config).to(device)
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

        out = solver.solve(x0=inp,
                           rhos=rgb_collim_model.rhos,
                           lams={reg_term: rgb_collim_model.sigmas.sqrt()},
                           max_iter=max_iter)
        return gt, inp, out

    print('Model size (M)', tlnn.benchmark.model_size(rgb_collim_model))
    print('Trainable model size (M)', tlnn.benchmark.trainable_model_size(rgb_collim_model))

    return solver, rgb_collim_model, step_fn


@torch.no_grad()
def test(
    savedir='saved/train_deconv',
    dataset='data/test/Urban100',
    checkpoint='best.pth',
    result_dir='results',
):
    _, rgb_collim_model, step_fn = build_model()

    # --------------- Load and Run ------------------ #
    savedir = Path(savedir)

    ckpt = torch.load(savedir / checkpoint)
    rgb_collim_model.load_state_dict(ckpt['model'])

    tl.metrics.set_data_format('chw')

    root = dataset
    dataset = os.path.basename(root)
    logger = tl.logging.Logger(savedir / result_dir / dataset, name=dataset)
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

        # logger.info('{} PSNR {} SSIM {}'.format(name, psnr, ssim))
        logger.save_img(name, pred)
        logger.save_img(name + '_inp.png', inp)
        logger.save_img(name + '_gt.png', gt)

    logger.info('averge results')
    logger.info(tracker.summary())
    print('Average Running Time', timer.toc() / len(os.listdir(root)))
    
    return tracker['psnr']


def train(
    root='data/bsd500',
    savedir='saved/train_deconv/',
    epochs=50,
    bs=2,
    lr=1e-4,
    resume=None,
):
    from dprox.utils.examples.optic.common import Dataset
    from torch.utils.data.dataloader import DataLoader
    import torch.nn.functional as F
    from tqdm import tqdm

    _, rgb_collim_model, step_fn = build_model()

    savedir = Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    logger = tl.logging.Logger(savedir)

    # ----------------- Start Training ------------------------ #
    dataset = Dataset(root)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    optimizer = torch.optim.AdamW(
        rgb_collim_model.parameters(),
        lr=1e-4, weight_decay=1e-3
    )
    tlnn.utils.adjust_learning_rate(optimizer, lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)

    epoch = 0
    gstep = 0
    best_psnr = 0
    imgdir = savedir / 'imgs'
    imgdir.mkdir(exist_ok=True, parents=True)

    psf = rgb_collim_model.get_psf()

    if resume:
        ckpt = torch.load(savedir / resume)
        rgb_collim_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        epoch = ckpt['epoch'] + 1
        gstep = ckpt['gstep'] + 1
        best_psnr = ckpt['best_psnr']

    while epoch < epochs:
        tracker = tl.trainer.util.MetricTracker()
        pbar = tqdm(total=len(loader), dynamic_ncols=True, desc=f'Epcoh[{epoch}]')
        for i, batch in enumerate(loader):
            if gstep % 10 == 0:
                psf = rgb_collim_model.get_psf()
                psf = crop_center_region(
                    normalize_psf(to_ndarray(psf, debatch=True), clip_percentile=0.01)
                )
                imageio.imsave(imgdir / f'psf_{gstep}.png', psf)
                
            gt, inp, pred = step_fn(batch)

            loss = F.mse_loss(gt, pred)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            psnr = tl.metrics.psnr(pred, gt)
            loss = loss.item()
            tracker.update('loss', loss)
            tracker.update('psnr', psnr)

            pbar.set_postfix({'loss': f'{tracker["loss"]:.4f}',
                              'psnr': f'{tracker["psnr"]:.4f}'})
            pbar.update()
                
            gstep += 1

        logger.info('Epoch {} Loss={} LR={}'.format(epoch, tracker['loss'], tlnn.utils.get_learning_rate(optimizer)))

        def save_ckpt(name):
            ckpt = {
                'model': rgb_collim_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'gstep': gstep,
                'psnr': tracker['psnr'],
                'best_psnr': best_psnr,
            }
            torch.save(ckpt, savedir / name)

        pbar.close()
        save_ckpt('last.pth')
        epoch += 1

        # validate
        with torch.no_grad():
            psnr = test(savedir=savedir, checkpoint='last.pth', result_dir='results_eval')
            if psnr > best_psnr:
                best_psnr = psnr
                save_ckpt('best.pth')
            logger.info('Validate Epoch {} PSNR={} Best PSNR={}'.format(epoch, psnr, best_psnr))


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'test': test,
    })
