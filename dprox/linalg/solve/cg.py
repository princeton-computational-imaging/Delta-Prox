from typing import Callable

import torch
import numpy as np


def bdot(x: torch.Tensor, y: torch.Tensor):
    """
    Perform batch dot product between two input tensors with the same shape.

    Args:
      x (torch.Tensor): The input tensor of the shape [batch_size, ...].
      y (torch.Tensor): Another input tensor with the same shape as x.

    Returns:
      Batch dot product of the shape [batch_size]
    """
    return torch.sum(x.reshape(x.shape[0], -1) * y.reshape(y.shape[0], -1), dim=-1)


def expand(x: torch.Tensor, ref: torch.Tensor):
    """
    Expands the input tensor to match the number of dimensions of the reference tensor.

    Args:
      x (torch.Tensor): The input tensor of any shape.
      ref (torch.Tensor): The reference tensor of any shape.

    Returns:
      the expanded tensor `x` to match the number of dimensions of the reference tensor `ref`.
    """
    while len(x.shape) < len(ref.shape):
        x = x.unsqueeze(-1)
    return x


def conjugate_gradient(
    A: Callable,
    b: torch.Tensor,
    x0: torch.Tensor = None,
    rtol: float = 1e-6,
    max_iters: int = 100,
    verbose: bool = False
):
    """
    Conjugate gradient method for solving a linear system of equations.

    Args:
      A (Callable): A is a callable function representing the forward operator A(x) of a matrix free linear operator.
      b (torch.Tensor): The parameter `b` is a tensor representing the right-hand side of the linear
        system of equations `Ax = b`.
      x0 (torch.Tensor): The initial guess for the solution vector. If not provided, it is initialized
        to a vector of zeros.
      rtol (float): Relative tolerance for convergence criteria. Default to 1e-6
      max_iters (int): The maximum number of iterations. Defaults to 100
      verbose (bool): Whether to logging intermediate information. Defaults to False

    Returns:
      The solution `x` to the linear system `Ax=b` using the conjugate gradient method.
    """

    # Temp vars
    x = torch.zeros_like(b)
    r = torch.zeros_like(b)
    Ap = torch.zeros_like(b)

    # Initialize x
    if x0 is not None:
        x = x0

    # Compute residual
    r = A(x)
    r *= - 1.0
    r += b

    cg_tol = rtol * torch.linalg.norm(b, 2, dim=(-1, -2))  # Relative tolerence

    # CG iteration
    gamma_1 = p = None
    cg_iter = np.minimum(max_iters, np.prod(b.shape))
    for iter in range(cg_iter):

        # Check for convergence
        normr = torch.linalg.norm(r, 2, dim=(-1, -2))
        if torch.all(normr <= cg_tol):
            if verbose:
                print("Converged at CG Iter %03d" % iter)
            break

        gamma = bdot(r, r)
        gamma = expand(gamma, x)

        # direction vector
        if iter > 0:
            beta = gamma / gamma_1
            p = r + beta * p
        else:
            p = r

        # Compute Ap
        Ap = A(p)

        # Cg update
        q = Ap

        tmp = bdot(p, q)
        alpha = gamma / expand(tmp, x)

        x = x + alpha * p  # update approximation vector
        r = r - alpha * q  # compute residual

        gamma_1 = gamma

    if verbose:
        print(f'Not converged, r norm={normr.tolist()}')

    return x


def conjugate_gradient2(
    A, b,
    x0=None, rtol=1e-6, max_iters=500, verbose=False
):
    # Solves A x = b
    x = torch.ones_like(b)

    if x0 is not None:
        print('use x init')
        x = x0

    r = b - A(x)
    d = r

    rnorm = r.ravel() @ r.ravel()
    for iter in range(max_iters):
        Ad = A(d)
        alpha = rnorm / (d.ravel() @ Ad.ravel())
        x = x + alpha * d
        r = r - alpha * Ad
        rnorm2 = r.ravel() @ r.ravel()
        beta = rnorm2 / rnorm
        rnorm = rnorm2
        d = r + beta * d
        if rnorm2 < rtol:
            if verbose: print(f'converge at iter={iter}, rtol={rtol}')
            break

    res = b - A(x)
    res = res.ravel() @ res.ravel()
    return x


def preconditioned_conjugate_gradient(
    A: Callable,
    b: torch.Tensor,
    Minv=None,
    x0: torch.Tensor = None,
    rtol: float = 1e-6,
    max_iters: int = 100,
    verbose: bool = False
):
    """
    Preconditioned conjugate gradient method for solving a linear system of equations. 
    The same as :func:conjugate_gradient except it could be preconditioned via `Minv`.

    Args:
      A (Callable): A is a callable function representing the forward operator A(x) of a matrix free linear operator.
      b (torch.Tensor): The parameter `b` is a tensor representing the right-hand side of the linear
        system of equations `Ax = b`.
      Minv (Callable):  A callable function representing the preconditioner.
      x0 (torch.Tensor): The initial guess for the solution vector. If not provided, it is initialized
        to a vector of zeros.
      rtol (float): Relative tolerance for convergence criteria. Default to 1e-6
      max_iters (int): The maximum number of iterations. Defaults to 100
      verbose (bool): Whether to logging intermediate information. Defaults to False

    Returns:
      The solution `x` to the linear system `Ax=b` using the conjugate gradient method.
    """

    if Minv is None:
        def Minv(x): return x

    if x0 is not None:
        x = x0
    else:
        x = torch.ones_like(b)

    r = A(x) - b
    y = Minv(r)
    p = - y

    bnorm = torch.linalg.vector_norm(b.ravel())

    for iter in range(max_iters):
        Ap = A(p)
        ry = r.ravel() @ y.ravel()
        alpha = ry / (p.ravel() @ Ap.ravel())
        x = x + alpha * p
        r = r + alpha * Ap
        y = Minv(r)
        # y = r
        beta = (r.ravel() @ y.ravel()) / ry
        p = - y + beta * p
        rnorm = torch.linalg.vector_norm(r.ravel())

        if rnorm < rtol * bnorm:
            break

    if verbose:
        print('#IT:', iter + 1)
    return x
