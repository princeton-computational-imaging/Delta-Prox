import torch
import os
import fire

import numpy as np
import imageio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Callable, Union, Iterable

from derain.data import get_test_data

from dprox.algo.base import auto_convert_to_tensor, move
from dprox import *
from dprox.utils import *
# from dprox.utils.examples.derain import LearnableDegOp

from dgunet_restormer import LearnableDegOp, unrolled_prior


def build_solver():
    # custom linop
    A = LearnableDegOp().cuda()
    def forward_fn(input, step): return A.forward(input, step)
    def adjoint_fn(input, step): return A.adjoint(input, step)
    raining = LinOpFactory(forward_fn, adjoint_fn)

    # build solver
    x = Variable()
    b = Placeholder()
    data_term = sum_squares(raining(x), b)
    reg_term = unrolled_prior(x)
    obj = data_term + reg_term
    solver = compile(obj, method='pgd', device='cuda')

    # load parameters
    ckpt = torch.load('derain_pdg_restormer.pth')
    A.load_state_dict(ckpt['linop'])
    reg_term.load_state_dict(ckpt['prior'])
    rhos = ckpt['rhos']

    return solver, rhos, b


@torch.no_grad()
def main(
    dataset='Rain100H',
    result_dir='results/DGUNet_dprox',
    data_dir='datasets/test/',
):
    solver, rhos, b = build_solver()

    rgb_dir_test = os.path.join(data_dir, dataset, 'input')
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)

    result_dir = os.path.join(result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)
    
    
    from derain.restormer import Restormer
    restormer = Restormer().cuda()
    restormer.load_state_dict(torch.load('restormer.pth')['params'])
    
    for data in tqdm(test_loader):
        input = data[0].cuda()
        filenames = data[1]

        def init_hook(iter, state, rho, lam):
            if iter == 5:
                state[0] = restormer(input)
        
        b.value = input
        output = solver.solve(x0=input, rhos=rhos, max_iter=7, callback=init_hook)
        output = output + input

        output = torch.clamp(output, 0, 1)
        output = output.permute(0, 2, 3, 1).cpu().detach().numpy() * 255

        for batch in range(len(output)):
            restored_img = output[batch].astype('uint8')
            imageio.imsave(
                os.path.join(result_dir, filenames[batch] + '.png'),
                restored_img
            )


if __name__ == '__main__':
    fire.Fire(main)
