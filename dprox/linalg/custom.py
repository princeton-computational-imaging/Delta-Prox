from dataclasses import dataclass, field
from functools import partial

import torch

from .solve import SOLVERS


@dataclass
class LinearSolveConfig:
    """ Defines default configuration parameters for solving linear equations.
    
    Args:
      rtol (float): The relative tolerance level for convergence, default to 1e-6.
      max_iters (int): The maximum number of iterations allowed for convergence.
      verbose (bool): whether to print progress updates during the solving process.
      solver_type (str): The type of solver to use (e.g. conjugate gradient).
      solver_kwargs (dict): additional keyword arguments to pass to the solver function 
    """
    rtol: float = 1e-6
    max_iters: int = 100
    verbose: bool = False
    solver_type: str = 'cg'
    solver_kwargs: dict = field(default_factory=dict)


def _build_solver(config: LinearSolveConfig):
    solve_fn = SOLVERS[config.solver_type]
    solve_fn = partial(solve_fn,
                       rtol=config.rtol,
                       max_iters=config.max_iters,
                       verbose=config.verbose,
                       **config.solver_kwargs)
    return solve_fn


def _trainable_parameters(module):
    return [p for p in module.parameters() if p.requires_grad]


class LinearSolve(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, b, config, *Aparams):
        ctx.A = A
        ctx.linear_solver = _build_solver(config)
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
        grad_Aparams = torch.autograd.grad((loss,), _trainable_parameters(A),
                                           grad_outputs=(grad_B,),
                                           create_graph=torch.is_grad_enabled(),
                                           allow_unused=True)

        return (None, grad_B, None, *grad_Aparams)


def linear_solve(A: torch.nn.Module, b: torch.Tensor, config: LinearSolveConfig = LinearSolveConfig()):
    """ Solves a linear system of equations with analytic gradient.

    Args:
      A (torch.nn.Module): A is a torch.nn.Module object, it should be callable as A(x) for forward operator 
        of the linear operator.
      b (torch.Tensor): b is a tensor representing the right-hand side of the linear system of equations Ax = b.
      config (LinearSolveConfig): `config` is an instance of the `LinearSolveConfig` class, which 
        contains various configuration options for the linear solver. These options include the maximum 
        number of iterations, the tolerance level for convergence, and the method used to solve the linear system.

    Returns:
      The solution of Ax = b.
    """
    return LinearSolve.apply(A, b, config, *_trainable_parameters(A))


def pcg(A: torch.nn.Module, b: torch.Tensor, rtol: float=1e-6, max_iters:int = 100, verbose: bool = False, **kwargs):
    config = LinearSolveConfig(rtol=rtol, max_iters=max_iters, verbose=verbose, solver_kwargs=kwargs, solver_type='pcg')
    return LinearSolve.apply(A, b, config, *_trainable_parameters(A))
