import sys
from pathlib import Path
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchlight as tl
import torchlight.nn as tlnn
import torchvision.utils
from common import (Dataset, build_doe_model, circular, device,
                    load_sample_img, plot, sanity_check, subplot)
from doe_model import img_psf_conv
from torch.utils.data import DataLoader
from tqdm import tqdm

from dprox import *
from dprox import Variable
from dprox.linop.conv import conv_doe
from dprox.utils import *
from dprox.utils.init.mosaic import masks_CFA_Bayer


def mosaicing(img):
    shape = img.shape[-2:]
    R_m, G_m, B_m = masks_CFA_Bayer(shape)
    mask = np.concatenate((R_m[..., None], G_m[..., None], B_m[..., None]), axis=-1)
    mask = torch.from_numpy(mask).to( img.device).permute(2,0,1).unsqueeze(0)
    b = mask * img
    return b


def main(args):
    savedir = Path(args.savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    logger = tl.logging.Logger(savedir)

    rgb_collim_model = build_doe_model()
    
    # -------------------- Define Model --------------------- #
    x = Variable()
    y = Placeholder()
    PSF = Placeholder()
    data_term = weighted_sum_squares(conv_doe(x, PSF, circular=circular), mosaic(x), y)
    reg_term = deep_prior(x, denoiser='ffdnet_color', trainable=False)
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
    sigma = args.sigma / 255.
    trainable_params = args.trainable_params
    rhos, sigmas = log_descent(49, 7.65, max_iter, sigma=max(0.255 / 255, sigma))
    rhos = torch.tensor(rhos, device=device).float()
    sigmas = torch.tensor(sigmas, device=device).float()
    rgb_collim_model.rhos = nn.parameter.Parameter(rhos, requires_grad=trainable_params)
    rgb_collim_model.sigmas = nn.parameter.Parameter(sigmas, requires_grad=trainable_params)
    rgb_collim_model.sigmas2 = nn.parameter.Parameter(sigmas, requires_grad=trainable_params)

    # ---------------- Forward Model ------------------------ #
    def step_fn(gt):
        gt = gt.to(device).float()
        psf = rgb_collim_model.get_psf()
        inp = img_psf_conv(gt, psf, circular=circular)
        inp = mosaicing(inp)
        inp = inp + torch.randn(*inp.shape, device=inp.device) * sigma
        y.value = inp
        PSF.value = psf
        out = solver.solve(x0=inp,
                           rhos=rgb_collim_model.rhos,
                           lams={reg_term: rgb_collim_model.sigmas, data_term: rgb_collim_model.sigmas2},
                           max_iter=max_iter)
        return gt, inp, out

    # ----------------- Sanity Check ------------------------ #
    gt, ob = sanity_check(rgb_collim_model.get_psf())
    gt, inp, out = step_fn(gt)
    logger.info(tl.metrics.psnr(out, gt))  # 38.55
    imshow(gt, ob, out)

    logger.info('Model size (M) {}'.format(tlnn.benchmark.model_size(rgb_collim_model)))
    logger.info('Trainable model size (M) {}'.format(tlnn.benchmark.trainable_model_size(rgb_collim_model)))

    # ----------------- Start Training ------------------------ #
    dataset = Dataset(args.root)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

    optimizer = torch.optim.AdamW(
        rgb_collim_model.parameters(),
        lr=1e-5, weight_decay=1e-3
    )
    tlnn.utils.adjust_learning_rate(optimizer, args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)

    epoch = 0
    gstep = 0
    best_psnr = 0
    imgdir = savedir / 'imgs'
    imgdir.mkdir(exist_ok=True, parents=True)

    psf = rgb_collim_model.get_psf()
    subplot(psf, imgdir / 'psf_init.png')

    if args.resume:
        ckpt = torch.load(savedir / args.resume)
        rgb_collim_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        epoch = ckpt['epoch'] + 1
        gstep = ckpt['gstep'] + 1
        best_psnr = ckpt['best_psnr']

    NUM_ACCUMULATES = 0
    while epoch < args.epochs:
        tracker = tl.trainer.util.MetricTracker()
        pbar = tqdm(total=len(loader), dynamic_ncols=True, desc=f'Epcoh[{epoch}]')
        for i, batch in enumerate(loader):
            gt, inp, pred = step_fn(batch)

            loss = F.mse_loss(gt, pred)
            
            if NUM_ACCUMULATES > 0:
                (loss / NUM_ACCUMULATES).backward()
                pred.detach_()
                loss.detach_()
                if i % NUM_ACCUMULATES == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                norm = 0
            else:
                loss.backward()
                norm = nn.utils.clip_grad_norm(rgb_collim_model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
    
            psnr = tl.metrics.psnr(pred, gt)
            loss = loss.item()
            tracker.update('loss', loss)
            tracker.update('psnr', psnr)
            tracker.update('norm', norm)

            pbar.set_postfix({'loss': f'{tracker["loss"]:.4f}',
                              'psnr': f'{tracker["psnr"]:.4f}',
                              'norm': f'{tracker["norm"]:.4f}'})
            pbar.update()

            if (gstep + 1) % 50 == 0:
                torchvision.utils.save_image(inp, imgdir / f'inp_{gstep}.png')
                torchvision.utils.save_image(pred, imgdir / f'pred_{gstep}.png')
                torchvision.utils.save_image(gt, imgdir / f'gt_{gstep}.png')

                psf = rgb_collim_model.get_psf()
                subplot(psf, imgdir / f'psf_{gstep}.png')

                height_map = rgb_collim_model.height_map.height_map_sqrt.detach()**2
                plot(height_map, imgdir / f'height_map_{gstep}.png')

                phase = torch.angle(rgb_collim_model.height_map.get_phase_profile())
                subplot(phase, imgdir / f'phase_{gstep}.png')
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
        # scheduler.step()

        # validate
        with torch.no_grad():
            img = load_sample_img()
            gt, inp, pred = step_fn(img)
            psnr = tl.metrics.psnr(pred, gt)
            if psnr > best_psnr:
                best_psnr = psnr
                save_ckpt('best.pth')
            logger.info('Validate Epoch {} PSNR={} Best PSNR={}'.format(epoch, psnr, best_psnr))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/bsd500')
    parser.add_argument('--savedir', type=str, default='saved/train_jd3_analytic/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('-tp', '--trainable_params', action='store_true')
    parser.add_argument('-r', '--resume', type=str, default=None)
    args = parser.parse_args()
    return args


main(parse_args())
