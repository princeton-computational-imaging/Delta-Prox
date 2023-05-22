from dataclasses import dataclass, field
from functools import partial

import torch
import dprox

from .solve import SOLVERS


@dataclass
class LinearSolveConfig:
    rtol: float = 1e-6
    max_iters: int = 100
    verbose: bool = False
    solver_type: str = 'cg'
    solver_kwargs: dict = field(default_factory=dict)


def build_solver(config: LinearSolveConfig):
    solve_fn = SOLVERS[config.solver_type]
    solve_fn = partial(solve_fn,
                       rtol=config.rtol,
                       max_iters=config.max_iters,
                       verbose=config.verbose,
                       **config.solver_kwargs)
    return solve_fn


def trainable_parameters(module):
    return [p for p in module.parameters() if p.requires_grad]


class LinearSolve(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, b, config, *Aparams):
        ctx.A = A
        ctx.linear_solver = build_solver(config)
        x = ctx.linear_solver(A, b)
        ctx.save_for_backward(x, *Aparams)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_B = ctx.linear_solver(ctx.A.T, grad_x)

        x = ctx.saved_tensors[0]
        x = x.detach().clone()

        A = ctx.A.clone()
        with torch.enable_grad():
            loss = -A(x)
        grad_Aparams = torch.autograd.grad((loss,), trainable_parameters(A),
                                           grad_outputs=(grad_B,),
                                           create_graph=torch.is_grad_enabled(),
                                           allow_unused=True)

        return (None, grad_B, None, *grad_Aparams)


def linear_solve(A: dprox.LinOp, b: torch.Tensor, config: LinearSolveConfig = LinearSolveConfig()):
    return LinearSolve.apply(A, b, config, *trainable_parameters(A))
