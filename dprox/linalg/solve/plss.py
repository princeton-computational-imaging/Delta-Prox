"""PLSS: A PROJECTED LINEAR SYSTEMS SOLVER 
For general linear system Ax = b with arbitrary matrix shape (supporting m = n; m > n; m < n)
PLSS and PLSSW are good at solving well-conditioned and ill-conditioned systems respectively.
See https://epubs.siam.org/doi/10.1137/22M1509783"""
import torch
from dprox import LinOp


def PLSS(A: LinOp, b, x0=None, rtol=1e-6, max_iters=500, verbose=False):
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
    
    rk = A(xk) -b
    rk_norm = torch.linalg.vector_norm(rk)
        
    if rk_norm_min < rk_norm:
        rk_norm = rk_norm_min
        xk = x_min
    
    if verbose:
        print(k + 1)
    return xk#, rk_norm


def PLSSW(A: LinOp, b, x0, Wh, rtol=1e-6, max_iters=500, verbose=False):
    Whi = 1 / Wh
    Whi[torch.isinf(Whi)] = 0
    
    xk = x
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
    return  xk#, nck
