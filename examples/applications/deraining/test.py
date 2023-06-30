import torch

from dprox import *
from dprox.utils import *
from dprox.utils.examples.derain import LearnableDegOp

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
solver = compile(obj, method='pgd')

# load parameters
ckpt = torch.load(get_path('checkpoints/derain_pdg.pth'))
A.load_state_dict(ckpt['linop'])
reg_term.load_state_dict(ckpt['prior'])
rhos = ckpt['rhos']

with torch.no_grad():
    out = solver.solve(x0=b, rhos=rhos, max_iter=7)
out = to_ndarray(out, debatch=True) + b
