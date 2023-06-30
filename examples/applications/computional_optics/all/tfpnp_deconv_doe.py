import torch
from scipy import ndimage

from dprox import *
from dprox.utils import *
from dprox.algo.tune import *

psf = point_spread_function(15, 5)


def build_solver():
    x = Variable()
    y = Placeholder()
    data_term = sum_squares(mosaic(conv(x, psf)), y)
    # data_term = weighted_sum_squares(conv(x, psf), mosaic(x), y)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    prob = Problem(data_term + reg_term)
    solver = compile(prob.objective, method='admm',
                     linear_solve_config=dict(
                         lin_solver_type='custom',
                         num_iters=500,
                         tol=1e-5,
                         verbose=True,
                     ))
    return solver, {'y': y}


def degrade(gt):
    y = ndimage.filters.convolve(gt, psf, mode='wrap')
    y = mosaicing_np(y)
    return y, {'y': y}


def main():
    solver, placeholders = build_solver()

    Dataset = DatasetFactory(degrade)
    dataset = Dataset('data/bsd500', target_size=128)
    valid_datasets = {
        'Kodak24': Dataset('data/test/Kodak24', size=1),
        'McMaster': Dataset('data/test/McMaster', size=1),
    }

    action_range = OrderedDict()
    action_range['rho'] = {'scale': 1e-5, 'shift': 0}
    for fn in solver.psi_fns:
        action_range[fn] = {'scale': 70 / 255, 'shift': 0}

    tf_solver = AutoTuneSolver(solver, policy='resnet', action_range=action_range)
    tf_solver.train(dataset, valid_datasets, placeholders, log_dir='tfpnp/jd2_500', validate_interval=10)
    tf_solver.eval('tfpnp/jd2_500/ckpt/actor_best.pkl', valid_datasets, placeholders)


if __name__ == '__main__':
    main()
