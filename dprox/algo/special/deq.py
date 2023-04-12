import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from dprox.algo.base import Algorithm
from dprox.utils import to_torch_tensor

from .deq_utils.jacobian import jac_loss_estimate
from .deq_utils.solvers import anderson, broyden


class DEQ(nn.Module):
    def __init__(self, fn, f_thres=40, b_thres=40):
        super().__init__()
        self.fn = fn

        self.f_thres = f_thres
        self.b_thres = b_thres

        self.solver = anderson
        # self.solver = broyden

    def forward(self, x, *args, **kwargs):
        f_thres = kwargs.get('f_thres', self.f_thres)
        b_thres = kwargs.get('b_thres', self.b_thres)

        z0 = x

        # Forward pass
        with torch.no_grad():
            out = self.solver(lambda z: self.fn(z, x, *args), z0, threshold=f_thres)
            z_star = out['result']   # See step 2 above
            new_z_star = z_star

        # (Prepare for) Backward pass, see step 3 above
        if self.training:
            new_z_star = self.fn(z_star.requires_grad_(), x, *args)

            # Jacobian-related computations, see additional step above. For instance:
            # jac_loss = jac_loss_estimate(new_z_star, z_star, vecs=1)

            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()   # To avoid infinite recursion
                # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
                out = self.solver(lambda y: torch.autograd.grad(new_z_star, z_star, y, retain_graph=True)[0] + grad,
                                  torch.zeros_like(grad), threshold=b_thres)
                new_grad = out['result']
                return new_grad

            self.hook = new_z_star.register_hook(backward_hook)

        return new_z_star


class DEQSolver(nn.Module):
    def __init__(self, solver: Algorithm, learned_params=False, rhos=None, lams=None):
        super().__init__()
        self.internal = solver

        def fn(z, x, *args):
            state = solver.unpack(z)
            state = solver.iter(state, *args)
            return solver.pack(state)
        self.solver = DEQ(fn)

        self.learned_params = learned_params
        if learned_params:
            self.r = nn.parameter.Parameter(torch.tensor(1.))
            self.l = nn.parameter.Parameter(torch.tensor(1.))
        
        self.rhos = rhos
        self.lams = lams

    def forward(self, x0, rhos, lams, **kwargs):
        lam = {k: to_torch_tensor(v)[..., 0].to(x0.device) for k, v in lams.items()}
        rho = to_torch_tensor(rhos)[..., 0].to(x0.device)

        if self.learned_params:
            rho = self.r * rho
            lam = {k: v * self.l for k, v in lam.items()}

        state = self.internal.initialize(x0)
        state = self.internal.pack(state)
        state = self.solver(state, rho, lam)
        state = self.internal.unpack(state)
        return state[0]

    def solve(self, x0, rhos, lams, **kwargs):
        return self.forward(x0, rhos, lams, **kwargs)

    def load(self, state_dict, strict=True):
        self.load_state_dict(state_dict['solver'], strict=strict)
        self.rhos = state_dict.get('rhos')
        self.lams = state_dict.get('lams')
        
    
def train_deq(args, dataset, solver, step_fn, on_epoch_end=None):
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    import torchlight as tl
    from torchlight.nn.utils import adjust_learning_rate
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
