from dataclasses import dataclass, field
from functools import partial

import torch
import dprox

from .solve import SOLVERS


@dataclass
class LinearSolveConfig:
    tol: float = 1e-5
    max_iters: int = 100
    verbose: bool = False
    solver_type: str = 'cg'
    solver_kwargs: dict = field(default_factory=dict)


def build_solver(config: LinearSolveConfig):
    solve_fn = SOLVERS[config.solver_type]
    solve_fn = partial(solve_fn,
                       tol=config.tol,
                       max_iters=config.max_iters,
                       verbose=config.verbose,
                       **config.solver_kwargs)
    return solve_fn


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
        Aparams = ctx.saved_tensors[1:]

        Aparams = [Aparam.clone().requires_grad_() for Aparam in Aparams]
        A = ctx.A.clone(Aparams)
        with torch.enable_grad():
            loss = -A(x)
        grad_Aparams = torch.autograd.grad((loss,), Aparams,
                                           grad_outputs=(grad_B,),
                                           create_graph=torch.is_grad_enabled(),
                                           allow_unused=True)

        return (None, grad_B, None, *grad_Aparams)


def linear_solve(A: dprox.LinOp, b: torch.Tensor, config: LinearSolveConfig = LinearSolveConfig()):
    out = LinearSolve.apply(A, b, config, *A.params)
    return out
