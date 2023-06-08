"""
PLSS: A PROJECTED LINEAR SYSTEMS SOLVER 
For general linear system Ax = b with arbitrary matrix shape (supporting m = n; m > n; m < n)
PLSS and PLSSW are good at solving well-conditioned and ill-conditioned systems respectively.
See https://epubs.siam.org/doi/10.1137/22M1509783
"""
from typing import Callable

import torch


def plss(
    A: Callable,
    b: torch.Tensor,
    x0: torch.Tensor = None,
    rtol: float = 1e-6,
    max_iters: int = 100,
    verbose: bool = False,
):
    """
    A Projective Linear Systems Solver (for well-conditioned system)

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
      The solution x to the linear system Ax=b, using the Preconditioned Least Squares Stationary iterative method.
    """
    if x0 is None: x0 = torch.zeros_like(b)
    x_min = xk = x0
    b_norm = torch.linalg.vector_norm(b)
    rk = A(xk) - b
    rk_norm_min = rk_norm = torch.linalg.vector_norm(rk)
    yk = A.adjoint(rk / rk_norm)

    rhok = rk_norm
    deltaki = 1 / torch.sum(yk * yk)
    pk = - (deltaki * rhok) * yk
    xk = xk + pk

    tol = rtol * b_norm
    for k in range(1, max_iters):
        rk = A(xk) - b
        rk_norm = torch.linalg.vector_norm(rk)

        # Store minimum iterate
        if rk_norm_min >= rk_norm:
            x_min = xk
            rk_norm_min = rk_norm

        if rk_norm <= tol:
            break

        yk = A.adjoint(rk / rk_norm)

        rhok = rk_norm
        p2 = torch.sum((pk * pk))
        nrp = torch.sqrt(p2)
        py = torch.sum((pk * yk))
        yy = torch.sum((yk * yk))
        ny = torch.sqrt(yy)

        denom = (nrp * ny - py) * (nrp * ny + py)

        beta1 = (rhok * py) / denom
        beta2 = - (rhok * p2) / denom

        # Step computation
        pk = beta1 * pk + beta2 * yk
        xk = xk + pk

    rk = A(xk) - b
    rk_norm = torch.linalg.vector_norm(rk)

    if rk_norm_min < rk_norm:
        rk_norm = rk_norm_min
        xk = x_min

    if verbose:
        print(k + 1)
    return xk  # , rk_norm


def plssw(
    A: Callable,
    b: torch.Tensor,
    Wh: torch.Tensor,
    x0: torch.Tensor = None,
    rtol: float = 1e-6,
    max_iters: int = 100,
    verbose: bool = False,
):
    """
    A Projective Linear Systems Solver Weighted (for ill-conditioned system)

    Args:
      A (Callable): A is a callable function representing the forward operator A(x) of a matrix free linear operator.
      b (torch.Tensor): The parameter `b` is a tensor representing the right-hand side of the linear
        system of equations `Ax = b`.
      Wh (torch.Tensor): A weight matrix used to adjust the importance of different components in the solution. It
        is used to compute the diagonal matrix Whi, which is the element-wise inverse of Wh
      x0 (torch.Tensor): The initial guess for the solution vector. If not provided, it is initialized
        to a vector of zeros.
      rtol (float): Relative tolerance for convergence criteria. Default to 1e-6
      max_iters (int): The maximum number of iterations. Defaults to 100
      verbose (bool): Whether to logging intermediate information. Defaults to False

    Returns:
      The solution x to the linear system Ax=b, using the Preconditioned Least Squares Stationary iterative method.
    """
    Whi = 1 / Wh
    Whi[torch.isinf(Whi)] = 0

    xk = x0
    ck = A(xk) - b
    nck = torch.linalg.vector_norm(ck)
    yk = A.adjoint(ck / nck)

    k = 0

    # Store minimum solution estimate
    xkmin = xk
    nckmin = nck

    rhok = nck
    zk = Whi * yk
    deltaki = 1 / torch.sum((zk * zk))
    pk = - (deltaki * rhok) * (Whi * zk)

    k = k + 1
    xk = xk + pk

    bnorm = torch.linalg.vector_norm(b)
    tol = rtol * bnorm

    for k in range(max_iters):
        Axk = A(xk)
        ck = Axk - b
        nck = torch.linalg.vector_norm(ck)

        # Store minimum iterate
        if nckmin >= nck:
            xkmin = xk
            nckmin = nck

        if nck <= tol:
            break

        yk = A.adjoint(ck / nck)
        zk = Whi * yk
        rhok = nck

        # Modifications for weighting
        Wp = Wh * pk
        p2 = torch.sum(Wp * Wp)

        nrp = torch.sqrt(p2)
        py = torch.sum((pk * yk))
        yy = torch.sum((zk * zk))
        ny = torch.sqrt(yy)

        denom = (nrp * ny - py) * (nrp * ny + py)
        beta1 = (rhok * py) / denom
        beta2 = - (rhok * p2) / denom

        # Step computation
        pk = beta1 * pk + beta2 * (Whi * zk)

        # Prepare for next iteration
        xk = xk + pk
        k = k + 1

    rk = A(xk) - b
    nck = torch.linalg.vector_norm(rk)

    if nckmin < nck:
        nck = nckmin
        xk = xkmin

    if verbose:
        print(k + 1)
    return xk  # , nck
