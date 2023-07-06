import torch
import os
import fire

import imageio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from derain.data import get_test_data

from dprox import *
from dprox.utils import *
# from dprox.utils.examples.derain import LearnableDegOp

from dgunet import LearnableDegOp, unrolled_prior


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
    # ckpt = torch.load(get_path('checkpoints/derain_pdg.pth'))
    # A.load_state_dict(ckpt['linop'])
    # reg_term.load_state_dict(ckpt['prior'])
    ckpt = torch.load('DGUNet_plus_conv_best.pth')

    state_dict = {}
    for k, v in ckpt['net'].items():
        if k.startswith('shallow_feat') and not k.startswith('shallow_feat1') and not k.startswith('shallow_feat7'):
            name = k[:len('shallow_feat1')]
            n = int(name[-1]) - 2
            k = k.replace(name, 'shallow_feat2')
            k = f'basic.{n}.' + k
        elif 'encoder' in k and 'stage1_encoder' not in k:
            name = k[:len('stage1')]
            n = int(name[-1]) - 2
            k = k.replace(name, 'stage2')
            k = f'basic.{n}.' + k
        elif 'decoder' in k and 'stage1_decoder' not in k:
            name = k[:len('stage1')]
            n = int(name[-1]) - 2
            k = k.replace(name, 'stage2')
            k = f'basic.{n}.' + k
        elif k.startswith('sam') and 'sam12' not in k:
            name = k[:len('sam12')]
            n = int(name[-1]) - 3
            k = k.replace(name, 'sam23')
            k = f'basic.{n}.' + k
        elif 'merge' in k and 'merge67' not in k:
            name = k[:len('merge12')]
            n = int(name[-1]) - 2
            k = k.replace(name, 'merge12')
            k = f'basic.{n}.' + k
            
        if not k.startswith('phi'):
            state_dict[k] = v
        
    state_dict2 = {}
    for k, v in ckpt['net'].items():
        if k.startswith('phi'):
            state_dict2[k] = v
    
    reg_term.denoiser.load_state_dict(state_dict, strict=False)
    A.load_state_dict(state_dict2, strict=False)
    
    rhos = torch.tensor([
        ckpt['net']['r0'],
        ckpt['net']['r1'],
        ckpt['net']['r2'],
        ckpt['net']['r3'],
        ckpt['net']['r4'],
        ckpt['net']['r5'],
        ckpt['net']['r6'],
    ])
    # rhos = ckpt['rhos']
    
    torch.save({'linop': A.state_dict(), 'prior': reg_term.state_dict(), 'rhos': rhos}, 'derain_pdg_unroll7.pth')

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

    for data in tqdm(test_loader):
        input = data[0].cuda()
        filenames = data[1]

        b.value = input
        output = solver.solve(x0=input, rhos=rhos, max_iter=7)
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
