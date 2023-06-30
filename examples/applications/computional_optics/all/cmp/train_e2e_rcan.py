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
                    load_sample_img, plot, sanity_check, build_baseline_profile, subplot,patch_size)
from doe_model import img_psf_conv
from torch.utils.data import DataLoader
from munch import munchify
from tqdm import tqdm

from dprox import *
from dprox import Variable
from dprox.linop.conv import conv_doe
from dprox.utils import *

from rcan import RCAN
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

    # -------------------- Define Model --------------------- #
    opt = dict(
        n_resgroups=4,
        n_resblocks=5,
        n_feats=32,
        rgb_range=1,
        n_colors=3,
        res_scale=1,
        reduction=16,
        scale=[1],
    )
    opt = munchify(opt)
    solver = RCAN(opt).to(device)
    rgb_collim_model = build_doe_model()

    # ---------------- Setup Hyperparameter ----------------- #
    sigma = args.sigma / 255.

    # ---------------- Forward Model ------------------------ #
    def step_fn(gt):
        gt = gt.to(device).float()
        profile = build_baseline_profile(rgb_collim_model)
        psf = rgb_collim_model.get_psf(profile)
        inp = img_psf_conv(gt, psf, circular=circular)
        inp = inp + torch.randn(*inp.shape, device=inp.device) * sigma
        out = solver(inp)
        return gt, inp, out

    # ----------------- Sanity Check ------------------------ #
    logger.info('Model size (M) {}'.format(tlnn.benchmark.model_size(solver)))
    logger.info('Trainable model size (M) {}'.format(tlnn.benchmark.trainable_model_size(solver)))

    # ----------------- Start Training ------------------------ #
    dataset = Dataset(args.root, crop_size=(patch_size, patch_size))
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

    optimizer = torch.optim.AdamW(
        solver.parameters(),
        lr=1e-4, weight_decay=1e-3
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
        solver.load_state_dict(ckpt['solver'])
        optimizer.load_state_dict(ckpt['optimizer'])
        epoch = ckpt['epoch'] + 1
        gstep = ckpt['gstep'] + 1
        best_psnr = ckpt['best_psnr']

    while epoch < args.epochs:
        tracker = tl.trainer.util.MetricTracker()
        pbar = tqdm(total=len(loader), dynamic_ncols=True, desc=f'Epcoh[{epoch}]')
        for i, batch in enumerate(loader):
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

            if (gstep + 1) % 200 == 0:
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

        logger.info('Epoch {} {} LR={}'.format(epoch, tracker.summary(), tlnn.utils.get_learning_rate(optimizer)))

        def save_ckpt(name):
            ckpt = {
                'model': rgb_collim_model.state_dict(),
                'solver': solver.state_dict(),
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
            img = load_sample_img(keep_ratio=False, patch_size=752)
            gt, inp, pred = step_fn(img)
            psnr = tl.metrics.psnr(pred, gt)
            logger.info('Validate Epoch {} PSNR={} Best PSNR={}'.format(epoch, psnr, best_psnr))
            if psnr > best_psnr:
                best_psnr = psnr
                save_ckpt('best.pth')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/bsd500')
    parser.add_argument('--savedir', type=str, default='saved/train_e2e_rcan/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--sigma', type=float, default=7.65)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('-r', '--resume', type=str, default=None)
    args = parser.parse_args()
    return args


main(parse_args())
