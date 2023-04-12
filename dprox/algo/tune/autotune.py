import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from munch import munchify
from PIL import Image
from tfpnp.data.batch import Batch
from tfpnp.env import PnPEnv
from tfpnp.eval import Evaluator
from tfpnp.pnp import PnPSolver
from tfpnp.policy.network import ResNetActorBase
from tfpnp.policy.sync_batchnorm import DataParallelWithCallback
from tfpnp.trainer import MDDPGTrainer
from tfpnp.utils.misc import apply_recursive
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from dprox.utils import to_torch_tensor

default_config = dict(
    rmsize=480,
    max_episode_step=6,
    train_steps=15000,
    warmup=20,
    save_freq=1000,
    validate_interval=50,
    episode_train_times=10,
    env_batch=48,
    loop_penalty=0.05,
    discount=0.99,
    lambda_e=0.2,
    tau=0.001,
    output='log',
    action_pack=5,
)


def rand_crop(img, croph, cropw):
    h, w, _ = img.shape
    h1 = random.randint(0, h - croph)
    w1 = random.randint(0, w - cropw)
    return img[h1:h1 + croph, w1:w1 + cropw, :]


def lr_scheduler(step):
    if step < 10000:
        return {'critic': 3e-4, 'actor': 1e-3}
    else:
        return {'critic': 1e-4, 'actor': 3e-4}


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def complex2channel(x):
    x = torch.view_as_real(x)
    N, C, H, W, _ = x.shape
    # N C H W 2 -> N 2 C H W
    x = x.permute(0, 1, 4, 2, 3).contiguous()
    x = x.view(N, C * 2, H, W)
    return x


POLICES = {
    'resnet': ResNetActorBase
}


def make_policy(solver, action_pack, ob_dim, type='resnet', action_range=None):
    policy = POLICES[type](ob_dim, action_pack, solver.nparams)

    if action_range is None:
        action_range = OrderedDict()
        for fn in solver.psi_fns:
            action_range[fn] = {'scale': 70 / 255, 'shift': 0}
        action_range['rho'] = {'scale': 1, 'shift': 0}

    policy.action_range = action_range
    return policy


class Env(PnPEnv):
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
        return torch.cat([
            variables,
            x0,
            ob.T,
        ], 1)

    def get_eval_ob(self, ob):
        return self.get_policy_ob(ob)

    def _get_attribute(self, ob, key):
        if key == 'gt':
            return ob.gt
        elif key == 'output':
            return self.solver.get_output(ob.variables)
        elif key == 'input':
            return ob.x0
        elif key == 'solver_input':
            return ob.variables, ob.params
        else:
            raise NotImplementedError('key is not supported, ' + str(key))

    def _build_next_ob(self, ob, solver_state):
        return Batch(gt=ob.gt,
                     x0=ob.x0,
                     y0=ob.y0,
                     params=ob.params,
                     variables=solver_state,
                     T=ob.T + 1 / self.max_episode_step)

    def _observation(self):
        idx_left = self.idx_left
        params = apply_recursive(lambda x: x[idx_left, ...], self.state['params'])
        ob = Batch(gt=self.state['gt'][idx_left, ...],
                   x0=self.state['x0'][idx_left, ...],
                   y0=self.state['y0'][idx_left, ...],
                   params=params,
                   variables=self.state['solver'][idx_left, ...],
                   T=self.state['T'][idx_left, ...])
        return ob


class TFPnPSolver(PnPSolver):
    def __init__(self, solver, solver_params):
        super().__init__(None)
        self.solver = solver
        self.solver_params = solver_params

        self.state_dim = solver.state_dim

    def forward(self, inputs, parameters, iter_num=None):
        variables, additional_params = inputs
        rhos, weights = parameters

        # infer iter_num from provided hyperparameters
        if iter_num is None:
            iter_num = rhos.shape[-1]

        for k in self.solver_params:
            self.solver_params[k].value = additional_params[k]

        states = self.solver.iters(self.solver.unpack(variables),
                                   rhos=rhos,
                                   lams=weights,
                                   max_iter=iter_num)

        return self.solver.pack(states)

    def reset(self, data):
        x0 = data['x0'].clone().detach()  # [B,1,W,H,2]
        for k in self.solver_params:
            self.solver_params[k].value = data['params'][k]
        states = self.solver.initialize(x0)
        return self.solver.pack(states)

    def get_output(self, state):
        return self.solver.unpack(state)[0].real

    def filter_hyperparameter(self, action):
        rhos = action['rho']
        weights = {k: v for k, v in action.items() if k not in ['idx_stop', 'rho']}
        return rhos, weights

    def filter_aux_inputs(self, state):
        return state['params']


class ImageDataset(Dataset):
    def __init__(self, datadir, fns=None, size=None, target_size=None, repeat=1):
        super().__init__()
        self.datadir = datadir
        self.fns = fns or [im for im in os.listdir(self.datadir) if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png") or im.endswith(".tif")]
        self.fns = sorted(self.fns)
        self.size = size
        self.repeat = repeat
        self.target_size = target_size

    def __getitem__(self, index):
        index = index % len(self.fns)
        imgpath = os.path.join(self.datadir, self.fns[index])

        target = Image.open(imgpath)
        if self.target_size is not None:
            target = target.resize((self.target_size, self.target_size))

        target = np.array(target, dtype=np.float32) / 255.0

        if target.ndim == 2:
            target = np.expand_dims(target, -1)

        return target

    def __len__(self):
        if self.size is None:
            return len(self.fns) * self.repeat
        else:
            return self.size


class MatDataset(Dataset):
    def __init__(self, datadir, fns=None, size=None, target_size=None, repeat=1):
        super().__init__()
        self.datadir = datadir
        self.fns = fns or [im for im in os.listdir(self.datadir) if im.endswith(".mat")]
        self.fns = sorted(self.fns)
        self.size = size
        self.repeat = repeat
        self.target_size = target_size

    def __getitem__(self, index):
        index = index % len(self.fns)
        imgpath = os.path.join(self.datadir, self.fns[index])

        # from scipy.io import loadmat
        from hdf5storage import loadmat
        target = loadmat(imgpath)['gt'].astype('float32')
        target = target / target.max()

        if self.target_size is not None:
            target = rand_crop(target, self.target_size, self.target_size)

        return target

    def __len__(self):
        if self.size is None:
            return len(self.fns) * self.repeat
        else:
            return self.size


BaseDatasts = {
    'mat': MatDataset,
    'img': ImageDataset
}


def DatasetFactory(degrade, base_dataset='img'):
    BaseDatast = BaseDatasts[base_dataset]

    class Dataset(BaseDatast):
        def __getitem__(self, index):
            gt = super().__getitem__(index)
            x0, params = degrade(gt)
            for k, v in params.items():
                if len(v.shape) == 2:
                    v = np.expand_dims(v, -1)
                params[k] = v.transpose(2, 0, 1)

            x0 = x0.transpose(2, 0, 1)
            gt = gt.transpose(2, 0, 1)
            dic = {'y0': x0, 'x0': x0, 'gt': gt, 'output': x0, 'input': x0}
            dic['params'] = params
            return dic
    return Dataset


class AutoTuneSolver(nn.Module):
    def __init__(self, solver, policy='resnet', action_pack=5, max_episode_step=6, custom_policy_ob_pack_fn=None):
        super().__init__()
        self.solver = solver
        # TODO: compute ob_dim based on ckpt
        self.policy = make_policy(solver, action_pack, ob_dim=9, type=policy)
        self.policy.eval()
        self.max_episode_step = max_episode_step
        self.custom_policy_ob_pack_fn = custom_policy_ob_pack_fn

    def solve(self, x0, aux_state=None, pbar=False):
        x0 = x0.to(self.solver.device)
        state = self.solver.initialize(x0)
        for i in tqdm(range(self.max_episode_step), disable=not pbar):
            rhos, lams, idx_stop = self.estimate(state, i, x0, aux_state)
            state = self.solver.iters(state, rhos, lams, self.policy.action_bundle)
        return state[0]

    @torch.no_grad()
    def estimate(self, states, iter, b, aux_state=None):
        policy_ob = self.solver.pack(states)
        B, _, W, H = policy_ob.shape

        vars = torch.split(policy_ob, policy_ob.shape[1] // self.solver.state_dim, dim=1)
        vars = [v[:, 0:1, :, :] for v in vars]
        variables = torch.cat(vars, dim=1)
        x0 = to_torch_tensor(b, batch=True).to(policy_ob.device)
        x0 = x0[:, 0:1, :, :]

        T = torch.ones([B, 1, W, H], device=policy_ob.device).float() * iter / self.max_episode_step

        if self.custom_policy_ob_pack_fn is not None:
            policy_ob = self.custom_policy_ob_pack_fn(variables, x0, T, aux_state)
        else:
            policy_ob = torch.cat([variables,
                                   x0,
                                   T,
                                   ], dim=1).real

        policy_ob = policy_ob.to(policy_ob.device)
        self.policy = self.policy.to(policy_ob.device)
        action, _, _, _ = self.policy(policy_ob, None, False, None)
        rhos = action['rho']
        weights = {k: v for k, v in action.items() if k not in ['idx_stop', 'rho']}
        return rhos, weights, action['idx_stop']

    def train(self,
              dataset,
              valid_datasets,
              placeholders,
              log_dir='log',
              config=default_config,
              device='cuda',
              resume=None,
              gpu_ids=None,
              custom_env=None,
              **kwargs
              ):
        if custom_env is not None:
            env = custom_env
        else:
            env = Env

        config = munchify(config)
        config.update(kwargs)
        config.output = log_dir
        device = torch.device(device)
        log_dir = Path(log_dir)

        solver = TFPnPSolver(self.solver, placeholders)

        if gpu_ids is not None:
            solver = DataParallelWithCallback(solver, device_ids=gpu_ids)
        else:
            solver = solver.to(device)

        val_loaders = {k: DataLoader(v, batch_size=1, shuffle=False) for k, v in valid_datasets.items()}
        eval_env = env(None, solver, self.max_episode_step).to(device)
        evaluator = Evaluator(eval_env, val_loaders, log_dir / 'results')

        train_loader = DataLoader(dataset, batch_size=config.env_batch, shuffle=True, num_workers=4, drop_last=True)
        env = env(train_loader, solver, self.max_episode_step).to(device)

        trainer = MDDPGTrainer(config,
                               env,
                               self.policy,
                               lr_scheduler=lr_scheduler,
                               log_dir=log_dir,
                               evaluator=evaluator,
                               device=device)
        if resume:
            trainer.load_model(resume)
        trainer.train()

    def eval(self,
             ckpt_path,
             valid_datasets,
             placeholders,
             log_dir='log',
             device='cuda',
             custom_env=None,
             ):
        if custom_env is not None:
            env = custom_env
        else:
            env = Env

        log_dir = Path(log_dir)
        device = torch.device(device)
        solver = TFPnPSolver(self.solver, placeholders).to(device)
        val_loaders = {k: DataLoader(v, batch_size=1, shuffle=False) for k, v in valid_datasets.items()}
        eval_env = env(None, solver, self.max_episode_step).to(device)
        evaluator = Evaluator(eval_env, val_loaders, log_dir / 'results')
        policy = self.policy.to(device)
        policy.load_state_dict(torch.load(ckpt_path))
        evaluator.eval(policy, 1)

    def load(self, ckpt_path):
        self.policy.load_state_dict(torch.load(ckpt_path))
