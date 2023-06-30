# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from dprox import *
from dprox.utils import *
from custom_linop import LearnableDegOp

b = imread('sample_input.png')
img = imread('sample_target.png')
img = to_torch_tensor(img, batch=True).cuda()
b = to_torch_tensor(b, batch=True).cuda()
imshow(img, b)

# custom linop
A = LearnableDegOp().cuda()
def forward_fn(inputs, step): return [A.forward(inputs[0], step)]
def adjoint_fn(inputs, step): return [A.adjoint(inputs[0], step)]
raining = LinOpFactory(forward_fn, adjoint_fn)

# define problem
x = Variable()
data_term = sum_squares(raining(x), b)
reg_term = unrolled_prior(x)
obj = data_term + reg_term
solver = compile(obj, method='pgd')

# load trained parameters
ckpt = torch.load('derain_pdg.pth')
ckpt_linop = filter_ckpt('diff_fn.linop.', ckpt)
ckpt_prior = filter_ckpt('prox_fn.', ckpt)
A.load_state_dict(ckpt_linop, strict=True)
reg_term.load_state_dict(ckpt_prior, strict=False)

# solve
max_iter = 7
rhos = torch.load('rho.pth')['rho']

out = solver.solve(x0=b, rhos=rhos, lams={reg_term: rhos}, max_iter=max_iter)
out = out + b

print(psnr(out, img))  # 33.41
imshow(img, out)

# %%
