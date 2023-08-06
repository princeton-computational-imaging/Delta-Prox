import os
from pathlib import Path

import fire
import imageio
import torch
import torch.utils.data
import torchlight as tl
import torchlight.nn as tlnn

from dprox import *
from dprox.contrib.optic import (DOEModelConfig, U_Net, build_doe_model,
                                 img_psf_conv, load_sample_img, normalize_psf)
from dprox.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model():
    solver = U_Net(3, 3).to(device)
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
def eval(step_fn, result_dir, dataset):
    tl.metrics.set_data_format('chw')

    root = hf.download_dataset(dataset)
    dataset = os.path.basename(root)
    logger = tl.logging.Logger(os.path.join(result_dir, dataset), name=dataset)
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
    dataset='Urban100',
    checkpoint='computational_optics/joint_deepoptics_unet.pth',
    result_dir='results/e2e_optics_unet',
):
    _, rgb_collim_model, step_fn = build_model()
    ckpt = hf.load_checkpoint(checkpoint)
    rgb_collim_model.load_state_dict(ckpt['model'])

    return eval(step_fn, result_dir=result_dir, dataset=dataset)


def train(
    training_dataset='BSD500',
    savedir='saved/e2e_optics_unet/',
    epochs=50,
    bs=2,
    lr=1e-4,
    resume=None,
):
    import torch.nn.functional as F
    from torch.utils.data.dataloader import DataLoader
    from tqdm import tqdm

    from dprox.contrib.optic import Dataset

    print('Training on device', device)

    solver, rgb_collim_model, step_fn = build_model()

    savedir = Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    logger = tl.logging.Logger(savedir)

    # ----------------- Start Training ------------------------ #
    root = hf.download_dataset(training_dataset)
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
