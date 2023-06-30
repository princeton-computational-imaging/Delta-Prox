import torch
from scipy import ndimage

from dprox import *
from dprox.utils import *
from dprox.algo.tune import *

psf = point_spread_function(15, 5)


def build_solver():
    x = Variable()
    y = Placeholder(0)
    data_term = sum_squares(conv(x, psf) - y)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    prob = Problem(data_term + reg_term)
    solver = compile(prob.objective, method='admm')
    return solver, {'y': y}


def degrade(gt):
    y = ndimage.convolve(gt, psf, mode='wrap')
    return y, {'y': y}


def main():
    solver, placeholders = build_solver()

    Dataset = DatasetFactory(degrade)
    dataset = Dataset('data/bsd500', target_size=96)
    valid_dataset = Dataset('data/Kodak24')

    tf_solver = AutoTuneSolver(solver, policy='resnet')
    tf_solver.train(dataset, {'Kodak24': valid_dataset}, placeholders)
    tf_solver.eval('log/ckpt/actor_best.pkl', {'Kodak24': valid_dataset}, placeholders)

if __name__ == '__main__':
    main()

