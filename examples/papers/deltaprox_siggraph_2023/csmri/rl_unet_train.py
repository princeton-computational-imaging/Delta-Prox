from scipy.io import loadmat
from pathlib import Path
from tfpnp.utils.noise import GaussianModelD

from dprox import *
from dprox.algo.tune import *
from dprox.utils import *
from dprox.contrib.csmri import (CustomADMM, CustomEnv,
                                 EvalDataset, TrainDataset,
                                 seed_everything)


def build_solver():
    x = Variable()
    y = Placeholder()
    mask = Placeholder()
    data_term = csmri(x, mask, y)
    reg_term = deep_prior(x, denoiser='unet')
    solver = CustomADMM([reg_term], [data_term])
    return solver, {'y': y, 'mask': mask}


def main():
    seed_everything(1234)

    solver, placeholders = build_solver()

    # dataset
    data_dir = Path('data')
    mask_dir = Path('data/csmri/masks')
    train_root = data_dir / 'Images_128'

    sigma_ns = [5, 10, 15]
    sampling_masks = ['radial_128_2', 'radial_128_4', 'radial_128_8']

    noise_model = GaussianModelD(sigma_ns)
    masks = [loadmat(mask_dir / f'{sampling_mask}.mat').get('mask')
             for sampling_mask in sampling_masks]
    dataset = TrainDataset(train_root, fns=None, masks=masks, noise_model=noise_model)

    valid_datasets = {
        'Medical7_2020/radial_128_2/15': EvalDataset('data/csmri/Medical7_2020/radial_128_2/15'),
        'Medical7_2020/radial_128_4/15': EvalDataset('data/csmri/Medical7_2020/radial_128_4/15'),
        'Medical7_2020/radial_128_8/15': EvalDataset('data/csmri/Medical7_2020/radial_128_8/15'),
    }

    # train

    tf_solver = AutoTuneSolver(solver, policy='resnet')

    training_cfg = dict(
        rmsize=480,
        max_episode_step=30,
        train_steps=15000,
        warmup=20,
        save_freq=1000,
        validate_interval=10,
        episode_train_times=10,
        env_batch=48,
        loop_penalty=0.05,
        discount=0.99,
        lambda_e=0.2,
        tau=0.001,
        action_pack=1,
        log_dir='rl_unet',
        custom_env=CustomEnv,
    )
    tf_solver.train(dataset, valid_datasets, placeholders, **training_cfg)
    ckpt_path = 'ckpt/tfpnp_unet/actor_best.pkl'
    tf_solver.eval(ckpt_path, valid_datasets, placeholders, custom_env=CustomEnv)


if __name__ == '__main__':
    main()
