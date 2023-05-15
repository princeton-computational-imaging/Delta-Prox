import abc
import torch
import torch.nn as nn
import numpy as np
from typing import List
from tqdm import tqdm

from dprox.proxfn import ProxFn
from dprox.linop import CompGraph, vstack
from dprox.utils import to_torch_tensor


def expand(r):
    if len(r.shape) == 1:
        r = r.view(r.shape[0], 1, 1, 1)
    return r


def auto_convert_to_tensor(names, batchify):
    def outter_wrapper(fn):
        def wrapper(*args, **kwargs):
            for k, v in kwargs.items():
                if k in names:
                    if v is not None:
                        kwargs[k] = to_tensor(v, batch=k in batchify)
            return fn(*args, **kwargs)
        return wrapper
    return outter_wrapper


def move(*args, device):
    return [to_device(arg, device) for arg in args]


def to_device(x, device):
    if x is None: return None
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x.to(device)


def to_tensor(x, batch=False):
    if isinstance(x, dict):
        return {k: to_tensor(v, batch) for k, v in x.items()}
    return to_torch_tensor(x, batch)


def isscalar(x):
    return np.isscalar(x) or (isinstance(x, torch.Tensor) and len(x.shape) == 0)


class Algorithm(nn.Module):
    # class method

    @abc.abstractclassmethod
    def partition(cls, prox_fns: List[ProxFn]):
        return NotImplementedError

    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    # instance method

    def __init__(self, psi_fns: List[ProxFn], omega_fns: List[ProxFn]):
        super().__init__()
        self.psi_fns = nn.ModuleList(psi_fns)
        self.omega_fns = nn.ModuleList(omega_fns)
        self.K = CompGraph(vstack([fn.linop for fn in psi_fns]))

    @property
    def device(self):
        return next(self.parameters()).device

    @auto_convert_to_tensor(['x0', 'rhos', 'lams'], batchify=['x0'])
    def solve(self, x0=None, rhos=None, lams=None, max_iter=24, pbar=False):
        x0, rhos, lams, max_iter = self.defaults(x0, rhos, lams, max_iter)
        x0, rhos, lams = move(x0, rhos, lams, device=self.device)

        state = self.initialize(x0)
        state = self.iters(state, rhos, lams, max_iter, pbar)

        return state[0]

    def iters(self, state, rhos, lams, max_iter, pbar=False):
        for iter in tqdm(range(max_iter), disable=not pbar):
            rho = rhos[..., iter]
            lam = {k: v[..., iter] for k, v in lams.items()}
            self._notify_all_op_current_step(iter)
            state = self.iter(state, rho, lam)
        return state

    def _notify_all_op_current_step(self, step):
        for fn in self.psi_fns:
            fn.step = step
        for fn in self.omega_fns:
            fn.step = step

        def recursive_set_step(op, step):
            op.step = step
            for node in op.input_nodes:
                recursive_set_step(node, step)

        for op in [fn.linop for fn in self.psi_fns]:
            recursive_set_step(op, step)
        for op in [fn.linop for fn in self.omega_fns]:
            recursive_set_step(op, step)

    def iter(self, state, rho, lam):
        self.K.update_vars([state[0]])
        state = self._iter(state, rho, lam)
        self.K.update_vars([state[0]])
        return state

    @abc.abstractmethod
    def _iter(self, state, rho, lam):
        return NotImplementedError

    def _notify_all_op_current_step(self, step):
        for fn in self.psi_fns:
            fn.step = step
        for fn in self.omega_fns:
            fn.step = step

        def recursive_set_step(op, step):
            op.step = step
            for node in op.input_nodes:
                recursive_set_step(node, step)

        for op in [fn.linop for fn in self.psi_fns]:
            recursive_set_step(op, step)
        for op in [fn.linop for fn in self.omega_fns]:
            recursive_set_step(op, step)

    def defaults(self, x0=None, rhos=None, lams=None, max_iter=24):
        # TODO: initialize with defaults if not specified
        if rhos is None: rhos = 1e-5
        if lams is None: lams = 0.02

        if isscalar(rhos): rhos = to_tensor([rhos] * max_iter)
        if isscalar(lams): lams = {fn: to_tensor([lams] * max_iter) for fn in self.psi_fns}

        lams = {k: to_tensor([v] * max_iter) if isscalar(v) else v for k, v in lams.items()}
        return x0, rhos, lams, max_iter

    # ---------------------------------------------------------------------------- #
    #                  Helper functions for reinforcement learning                 #
    # ---------------------------------------------------------------------------- #

    def pack(self, state):
        flatten = []
        for s in state:
            if isinstance(s, list): flatten += s
            else: flatten += [s]
        return torch.cat(flatten, dim=1)

    #TODO: refactor
    def unpack(self, tensor):
        vars = list(torch.split(tensor,
                                tensor.shape[1] // self.state_dim,
                                dim=1))
        splits = []
        start, end = 0, 0
        for d in self.state_split:
            if d == 1:
                splits.append(vars[start])
                end += 1
            else:
                d = d[0]
                end += d
                splits.append(vars[start:end])
            start = end
        return splits

    @abc.abstractproperty
    def state_dim(self):
        ans = 0
        for s in self.state_split:
            if isinstance(s, list): ans += sum(s)
            else: ans += s
        return ans

    @abc.abstractproperty
    def nparams(self):
        return NotImplementedError

    @abc.abstractproperty
    def state_split(self):
        return NotImplementedError
