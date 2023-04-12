import torch

from dprox.algo.admm import ADMM
from dprox.algo.tune import Batch, Env, apply_recursive, complex2channel

from .dataset import CSMRIDataset, CSMRIEvalDataset


class TrainDataset(CSMRIDataset):
    def __getitem__(self, index):
        dic = super().__getitem__(index)
        dic['params'] = {'y': dic['y0'], 'mask': dic['mask']}
        return dic


class EvalDataset(CSMRIEvalDataset):
    def __getitem__(self, index):
        dic = super().__getitem__(index)
        dic['params'] = {'y': dic['y0'], 'mask': dic['mask']}
        return dic


class CustomADMM(ADMM):
    def _iter(self, state, rho, lam):
        x, z, u = state
        x = [x]
        z = z[0]

        for i, fn in enumerate(self.psi_fns):
            x[i] = fn.prox(z - u[i], lam=lam[fn])

        b = [x[i] + u[i] for i in range(len(self.psi_fns))]
        z = self.least_square.solve(b, rho)

        for i, fn in enumerate(self.psi_fns):
            u[i] = u[i] + x[i] - z

        return x[0], [z], u


class CustomEnv(Env):
    def __init__(self, data_loader, solver, max_episode_step):
        super().__init__(data_loader, solver, max_episode_step)
        self.channels = 0

    def get_policy_ob(self, ob):
        chs = self.channels
        vars = torch.split(ob.variables,
                           ob.variables.shape[1] // self.solver.state_dim, dim=1)
        vars = [v[:, chs:chs + 1, :, :] for v in vars]
        variables = torch.cat(vars, dim=1)
        x0 = ob.x0[:, chs:chs + 1, :, :]
        ob = torch.cat([
            variables,
            complex2channel(ob.y0),
            x0,
            ob.mask,
            ob.T,
            ob.sigma_n,
        ], 1).real
        return ob

    def _build_next_ob(self, ob, solver_state):
        return Batch(gt=ob.gt,
                     x0=ob.x0,
                     y0=ob.y0,
                     params=ob.params,
                     variables=solver_state,
                     mask=ob.mask,
                     sigma_n=ob.sigma_n,
                     T=ob.T + 1 / self.max_episode_step)

    def _observation(self):
        idx_left = self.idx_left
        params = apply_recursive(lambda x: x[idx_left, ...], self.state['params'])
        ob = Batch(gt=self.state['gt'][idx_left, ...],
                   x0=self.state['x0'][idx_left, ...],
                   y0=self.state['y0'][idx_left, ...],
                   params=params,
                   variables=self.state['solver'][idx_left, ...],
                   mask=self.state['mask'][idx_left, ...].float(),
                   sigma_n=self.state['sigma_n'][idx_left, ...],
                   T=self.state['T'][idx_left, ...])
        return ob


def custom_policy_ob_pack_fn(variables, x0, T, aux_state):
    return torch.cat([variables,
                      complex2channel(aux_state['y0']).cuda(),
                      x0,
                      aux_state['mask'].cuda(),
                      T,
                      aux_state['sigma_n'].cuda(),
                      ], dim=1).real
