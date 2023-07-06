import torch.nn as nn
import torch.utils.data
import torch

import os
from pathlib import Path

import imageio
import fire
import torch
import torch.nn as nn
import torchlight as tl
import torchlight.nn as tlnn

from dprox import *
from dprox.utils import *
from dprox.utils.examples.optic.common import (
    build_doe_model, normalize_psf, load_sample_img, DOEModelConfig
)
from dprox.utils.examples.optic.doe_model import img_psf_conv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))
        
        self.conv_residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = self.conv(x) + self.conv_residual(x)
        return x


class U_Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(U_Net, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Down1 = nn.Conv2d(filters[0], filters[0], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down2 = nn.Conv2d(filters[1], filters[1], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down3 = nn.Conv2d(filters[2], filters[2], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down4 = nn.Conv2d(filters[3], filters[3], kernel_size=4, stride=2, padding=1, bias=True)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Down1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Down2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Down3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Down4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out+x
    
def build_model():
    solver = U_Net(3,3).to(device)
    config = DOEModelConfig()
    rgb_collim_model = build_doe_model(config).to(device)
    circular = config.circular

    # ---------------- Setup Hyperparameter ----------------- #
    sigma = 7.65 / 255.

    # ---------------- Forward Model ------------------------ #
    def step_fn(gt):
        gt = gt.to(device).float()
        psf = rgb_collim_model.get_psf()
        inp = img_psf_conv(gt, psf, circular=circular)
        inp = inp + torch.randn(*inp.shape, device=inp.device) * sigma
        out = solver(inp)
        return gt, inp, out
    
    print('Model size (M)', tlnn.benchmark.model_size(rgb_collim_model))
    print('Trainable model size (M)', tlnn.benchmark.trainable_model_size(rgb_collim_model))

    return solver, rgb_collim_model, step_fn

@torch.no_grad()
def eval(step_fn, result_dir, savedir, dataset):
    savedir = Path(savedir)
    tl.metrics.set_data_format('chw')

    root = dataset
    dataset = os.path.basename(root)
    logger = tl.logging.Logger(savedir / result_dir / dataset, name=dataset)
    tracker = tl.trainer.util.MetricTracker()

    timer = tl.utils.Timer()
    timer.tic()
    for idx, name in enumerate(os.listdir(root)):
        gt = load_sample_img(os.path.join(root, name), patch_size=768)

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


def test(
    savedir='saved/train_deconv',
    dataset='data/test/Urban100',
    checkpoint='best.pth',
    result_dir='results',
):
    _, rgb_collim_model, step_fn = build_model()
    ckpt = torch.load(savedir / checkpoint)
    rgb_collim_model.load_state_dict(ckpt['model'])

    return eval(step_fn, result_dir=result_dir,
                dataset=dataset, savedir=savedir)


def train(
    root='data/bsd500',
    savedir='saved/e2e_unet/',
    epochs=50,
    bs=2,
    lr=1e-4,
    resume=None,
):
    from dprox.utils.examples.optic.common import Dataset
    from torch.utils.data.dataloader import DataLoader
    import torch.nn.functional as F
    from tqdm import tqdm

    print('Training on device', device)

    solver, rgb_collim_model, step_fn = build_model()

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
    optimizer2 = torch.optim.AdamW(
        solver.parameters(),
        lr=1e-4, weight_decay=1e-3
    )
    
    tlnn.utils.adjust_learning_rate(optimizer, lr)
    tlnn.utils.adjust_learning_rate(optimizer2, lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)

    epoch = 0
    gstep = 0
    best_psnr = 0
    imgdir = savedir / 'imgs'
    imgdir.mkdir(exist_ok=True, parents=True)

    if resume:
        ckpt = torch.load(savedir / resume)
        rgb_collim_model.load_state_dict(ckpt['model'])
        solver.load_state_dict(ckpt['solver'])
        optimizer.load_state_dict(ckpt['optimizer'])
        epoch = ckpt['epoch'] + 1
        gstep = ckpt['gstep'] + 1
        best_psnr = ckpt['best_psnr']

    def save_ckpt(name, psnr=0):
        ckpt = {
            'model': rgb_collim_model.state_dict(),
            'solver': solver.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'gstep': gstep,
            'psnr': psnr,
            'best_psnr': best_psnr,
        }
        torch.save(ckpt, savedir / name)

    save_ckpt('last.pth')
    history_psf = []
    history_psnr = []

    while epoch < epochs:
        tracker = tl.trainer.util.MetricTracker()
        pbar = tqdm(total=len(loader), dynamic_ncols=True, desc=f'Epcoh[{epoch}]')

        for i, batch in enumerate(loader):
            if gstep % 10 == 0:
                # validate
                with torch.no_grad():
                    psnr = eval(
                        step_fn=step_fn,
                        dataset='data/visual',
                        result_dir='results_eval',
                        savedir=savedir,
                    )
                    if psnr > best_psnr:
                        best_psnr = psnr
                        save_ckpt('best.pth')
                    logger.info('Validate Epoch {} PSNR={} Best PSNR={}'.format(epoch, psnr, best_psnr))

                psf = rgb_collim_model.get_psf()
                psf = crop_center_region(
                    normalize_psf(to_ndarray(psf, debatch=True), clip_percentile=0.01)
                )
                imageio.imsave(imgdir / f'psf_{gstep}.png', psf)
                history_psf.append(psf)
                history_psnr.append(psnr)

                with open(savedir / 'stat.txt', 'a') as f:
                    f.write(f'{gstep} {psnr}\n')

            gt, inp, pred = step_fn(batch)

            loss = F.mse_loss(gt, pred)
            loss.backward()

            optimizer.step()
            optimizer2.step()
            optimizer.zero_grad()
            optimizer2.zero_grad()

            psnr = tl.metrics.psnr(pred, gt)
            loss = loss.item()
            tracker.update('loss', loss)
            tracker.update('psnr', psnr)

            pbar.set_postfix({'loss': f'{tracker["loss"]:.4f}',
                              'psnr': f'{tracker["psnr"]:.4f}'})
            pbar.update()

            gstep += 1

        logger.info('Epoch {} Loss={} LR={}'.format(epoch, tracker['loss'], tlnn.utils.get_learning_rate(optimizer)))

        save_ckpt('last.pth', tracker['psnr'])
        pbar.close()
        epoch += 1

    imageio.mimsave(savedir / 'psf.mp4', history_psf)


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'test': test,
    })
