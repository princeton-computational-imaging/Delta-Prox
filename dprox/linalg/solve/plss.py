"""PLSS: A PROJECTED LINEAR SYSTEMS SOLVER 
For general linear system Ax = b with arbitrary matrix shape (supporting m = n; m > n; m < n)
PLSS and PLSSW are good at solving well-conditioned and ill-conditioned systems respectively.
See https://epubs.siam.org/doi/10.1137/22M1509783"""
import torch
from dprox import LinOp


def PLSS(A: LinOp, b, x0=None, rtol=1e-6, max_iters=500, verbose=False):
    """
    PLSS: A PROJECTED LINEAR SYSTEMS SOLVER (for well-conditioned system)
    
    :param A: A is a linear operator, which is a function that maps a vector to another vector in a
    linear space. In this context, it is used to represent a linear system of equations
    :type A: LinOp
    :param b: The right-hand side vector in the linear system Ax=b
    :param x0: Initial guess for the solution vector. If not provided, a vector of zeros with the same
    shape as b will be used
    :param rtol: Relative tolerance for convergence criteria. The algorithm stops iterating when the
    relative residual is below this value
    :param max_iters: The maximum number of iterations allowed for the algorithm to converge to a
    solution, defaults to 500 (optional)
    :param verbose: A boolean parameter that determines whether or not to print the number of iterations
    taken by the algorithm. If set to True, the function will print the number of iterations taken. If
    set to False, it will not print anything, defaults to False (optional)
    :return: the solution xk to the linear system Ax=b, where A is a linear
    operator and b is a vector, using the Preconditioned Least Squares Stationary iterative method.
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
    
    rk = A(xk) -b
    rk_norm = torch.linalg.vector_norm(rk)
        
    if rk_norm_min < rk_norm:
        rk_norm = rk_norm_min
        xk = x_min
    
    if verbose:
        print(k + 1)
    return xk#, rk_norm


def PLSSW(A: LinOp, b, x0, Wh, rtol=1e-6, max_iters=500, verbose=False):
    """
    PLSS: A PROJECTED LINEAR SYSTEMS SOLVER WEIGHTED (for ill-conditioned system)
    
    :param A: A is a linear operator, which is a function that maps one vector space to another. In this
    case, it is used to represent a linear system of equations
    :type A: LinOp
    :param b: The right-hand side vector in the linear system Ax=b
    :param x0: Initial guess for the solution vector
    :param Wh: A weight matrix used to adjust the importance of different components in the solution. It
    is used to compute the diagonal matrix Whi, which is the element-wise inverse of Wh
    :param rtol: Relative tolerance for convergence criteria. The algorithm stops when the norm of the
    residual is less than rtol times the norm of the right-hand side vector b
    :param max_iters: The maximum number of iterations allowed for the algorithm to converge to a
    solution, defaults to 500 (optional)
    :param verbose: A boolean parameter that determines whether or not to print out information during
    the algorithm's execution. If set to True, the algorithm will print out the number of iterations it
    took to converge. If set to False, it will run silently, defaults to False (optional)
    :return: the final solution estimate xk.
    """
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
