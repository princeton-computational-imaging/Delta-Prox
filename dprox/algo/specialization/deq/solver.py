from typing import Union

import torch
import torch.nn as nn
import numpy as np

from dprox.algo.base import Algorithm, move, auto_convert_to_tensor
from dprox.utils import to_torch_tensor

from .utils.solvers import anderson


class DEQ(nn.Module):
    def __init__(self, fn, f_thres=40, b_thres=40):
        super().__init__()
        self.fn = fn

        self.f_thres = f_thres
        self.b_thres = b_thres

        self.solver = anderson

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

    @auto_convert_to_tensor(['x0', 'rhos', 'lams'], batchify=['x0'])
    def solve(
        self,
        x0: Union[torch.Tensor, np.ndarray] = None,
        rhos: Union[float, torch.Tensor, np.ndarray] = None,
        lams: Union[float, torch.Tensor, np.ndarray, dict] = None,
        **kwargs
    ):
        x0, rhos, lams, _ = self.internal.defaults(x0, rhos, lams, 1)
        x0, rhos, lams = move(x0, rhos, lams, device=self.internal.device)
        
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

    def forward(
        self,
        **kwargs
    ):
        return self.solve(**kwargs)

    def load(self, state_dict, strict=True):
        self.load_state_dict(state_dict['solver'], strict=strict)
        self.rhos = state_dict.get('rhos')
        self.lams = state_dict.get('lams')
