import torch
import numpy as np
from functools import partial


def PCG(A, b, Minv=None, x_init=None, btol=1e-5, num_iters=100, verbose=False):
    if Minv is None:
        Minv = lambda x: x

    if x_init is not None:
        x = x_init
    else:
        x = torch.ones_like(b)
        
    r = A(x) - b
    y = Minv(r)
    # y = r
    p = - y

    bnorm = torch.linalg.norm(b)
    
    for iter in range(num_iters):
        Ap = A(p)
        ry = r.T @ y
        alpha = ry / (p.T @ Ap)
        x = x + alpha * p
        r = r + alpha * Ap
        y = Minv(r)
        # y = r
        beta = (r.T @ y) / ry
        p = - y + beta * p
        rnorm = torch.linalg.norm(r)
        
        if rnorm < btol * bnorm:
            break
    
    if verbose:
        print(iter + 1)
    return x


"""Projected linear systems solver"""
def PLSS(x0, A, b, atol=1e-6, btol=1e-6, maxiter=500, ord=2):
    m, n = A.shape
    x_min = xk = x0
    b_norm = torch.linalg.vector_norm(b, ord=ord)
    rk = A(xk) - b
    rk_norm_min = rk_norm = torch.linalg.vector_norm(rk, ord=ord)
    yk = A.adjoint(rk / rk_norm)
    
    rhok = rk_norm
    deltaki = 1 / torch.sum((yk * yk))
    pk = - (deltaki * rhok) * yk
    xk = xk + pk
    
    for k in range(1, maxiter):
        rk = A(xk) - b
        rk_norm = torch.linalg.vector_norm(rk, ord=ord)
        
        # Store minimum iterate
        if rk_norm_min >= rk_norm:            
            x_min = xk
            rk_norm_min = rk_norm
        
        if rk_norm <= m**0.5 * atol + btol * b_norm:
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
    rk_norm = torch.linalg.vector_norm(rk, ord=ord)
        
    if rk_norm_min < rk_norm:
        rk_norm = rk_norm_min
        xk = x_min
    
    # print(k + 1)
    # print(rk_norm / b_norm)
    return xk, rk_norm
