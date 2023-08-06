import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchlight as tl
from torch.utils.data import DataLoader
from torchlight.nn.utils import adjust_learning_rate
from tqdm import tqdm


def train_deq(args, dataset, solver, step_fn, on_epoch_end=None):
    from torch.utils.tensorboard import SummaryWriter

    loader = DataLoader(dataset, batch_size=args.bs, num_workers=8, shuffle=True)
    device = torch.device('cuda')

    optimizer = torch.optim.AdamW(solver.parameters(), lr=1e-4, weight_decay=1e-4)

    savedir = Path(args.savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(savedir / 'tf_log')
    gstep = 0
    start_epoch = 0

    if args.resume:
        path = os.path.join(args.savedir, args.resume)
        ckpt = torch.load(path)
        solver.load_state_dict(ckpt['solver'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        gstep = ckpt['gstep'] + 1

    adjust_learning_rate(optimizer, lr=1e-5)

    reset = False
    epoch = start_epoch

    solver.train()
    while epoch < args.epochs:
        if reset:
            ckpt = torch.load(savedir / 'last.pth')
            solver.load_state_dict(ckpt['solver'], strict=False)
            optimizer.load_state_dict(ckpt['optimizer'])
            gstep = ckpt['gstep']
            epoch = ckpt['epoch']
            reset = False

        total_psnr = 0
        pbar = tqdm(total=len(loader), dynamic_ncols=True, desc=f'Epcoh[{epoch}]')
        total_loss = 0
        prev_loss = None
        step = 0

        for idx, batch in enumerate(loader):
            optimizer.zero_grad()
            gt, pred = step_fn(batch)
            loss = F.mse_loss(pred, gt)

            # if gstep > 200 and (torch.isnan(loss) or (prev_loss and loss.item() > prev_loss * 5)):
            #     print('detect unnormal loss, ', loss.item(), prev_loss)
            #     reset = True
            #     break

            loss.backward()
            norm = 0
            # norm = nn.utils.clip_grad_norm_(solver.parameters(), max_norm=10).item()
            optimizer.step()

            psnr = tl.metrics.psnr(pred, gt)
            total_psnr += psnr
            total_loss += loss.item()
            step += 1
            prev_loss = loss.item()

            pbar.set_postfix({'loss': f'{total_loss / step:.8f}',
                              'psnr': f'{total_psnr / step:.4f}',
                              'norm': norm})
            pbar.update()

            gstep += 1
            writer.add_scalar('loss', loss.item(), gstep)
            writer.add_scalar('psnr', psnr, gstep)

            if (idx + 1) % 100 == 0:
                # print('save intermediate ckpt')
                prev_loss = loss.item()
                ckpt = {
                    'solver': solver.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'gstep': gstep,
                }
                torch.save(ckpt, savedir / 'last.pth')

        if not reset:
            # save things
            print('Save checkpoint')
            ckpt = {
                'solver': solver.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'gstep': gstep,
            }
            # if (epoch+1) % 5 == 0:
            torch.save(ckpt, savedir / f'epoch_{epoch}.pth')
            torch.save(ckpt, savedir / 'last.pth')
            epoch += 1

        pbar.close()

        if on_epoch_end is not None:
            on_epoch_end()
