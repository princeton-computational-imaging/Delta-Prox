import torch
import numpy as np

# TODO: standardize the stopping criterion


def conjugate_gradient(A, b, x_init=None, tol=1e-6, num_iters=500, verbose=False):
    # Solves A x = b with

    # Temp vars
    x = torch.zeros_like(b)
    r = torch.zeros_like(b)
    Ap = torch.zeros_like(b)

    # Initialize x
    # Initialize everything to zero.
    if x_init is not None:
        x = x_init

    # Compute residual
    # r = b - KtKfun(x)
    r = A(x)
    r *= -1.0
    r += b

    # Do cg iterations

    cg_tol = tol * torch.linalg.norm(b.ravel(), 2)  # Relative tol

    # CG iteration
    gamma_1 = p = None
    cg_iter = np.minimum(num_iters, np.prod(b.shape))
    for iter in range(cg_iter):
        # Check for convergence

        normr = torch.linalg.norm(r.ravel(), 2)

        # Check for convergence
        if normr <= cg_tol:
            # Iterate
            if verbose:
                print("Converged at CG Iter %03d" % iter)
            break

        gamma = torch.dot(r.ravel(), r.ravel())

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

        alpha = gamma / torch.dot(p.ravel(), q.ravel())

        x = x + alpha * p  # update approximation vector
        r = r - alpha * q  # compute residual

        gamma_1 = gamma

    return x


def bdot(A, B):
    return torch.bmm(A.unsqueeze(dim=1), B.unsqueeze(dim=2)).squeeze(1).squeeze(1)


def expand(x, ref):
    while len(x.shape) < len(ref.shape):
        x = x.unsqueeze(-1)
    return x


def conjugate_gradient2(A, b, x_init=None, tol=1e-5, num_iters=500, verbose=False):
    # Solves A x = b
    x = torch.ones_like(b)

    if x_init is not None:
        print('use x init')
        x = x_init

    r = b - A(x)
    d = r

    rnorm = r.ravel() @ r.ravel()
    for iter in range(num_iters):
        Ad = A(d)
        alpha = rnorm / (d.ravel() @ Ad.ravel())
        x = x + alpha * d
        r = r - alpha * Ad
        rnorm2 = r.ravel() @ r.ravel()
        beta = rnorm2 / rnorm
        rnorm = rnorm2
        d = r + beta * d
        if rnorm2 < tol:
            if verbose: print(f'converge at iter={iter}, tol={tol}')
            break

    res = b - A(x)
    res = res.ravel() @ res.ravel()
    return x


def conjugate_gradient3(A, b, x_init=None, tol=1e-7, num_iters=500, verbose=False):
    # Solves A x = b
    x = torch.ones_like(b)

    if x_init is not None:
        print('use x init')
        x = x_init

    r = b - A(x)
    d = r

    eps = 1e-7
    rnorm = bdot(r.flatten(1), r.flatten(1))
    # rnorm = (r * r).flatten(1).sum((1,2,3))
    # print(torch.allclose(rnorm, rnormg))
    for iter in range(num_iters):
        Ad = A(d)
        tmp = bdot(d.flatten(1), Ad.flatten(1))
        # tmp = (d * Ad).sum((1,2,3))
        
        # tmp = tmp + eps
        if tmp.min() < eps:
            tmp = tmp + eps
            
        alpha = rnorm / tmp
        alpha = expand(alpha, x)
        x = x + alpha * d
        r = r - alpha * Ad
        rnorm2 = bdot(r.flatten(1), r.flatten(1))
        # rnorm2 = (r * r).sum((1,2,3))
        
        if rnorm.min() < eps:
            rnorm = rnorm + eps
        # rnorm = rnorm + eps
          
        beta = rnorm2 / rnorm
        # if torch.isnan(beta).any():
            # import ipdb; ipdb.set_trace()
        rnorm = rnorm2
        beta = expand(beta, x)
        d = r + beta * d
        
        res = b - A(x)
        res = res.ravel() @ res.ravel()
        if res < tol:
        # if rnorm2.max() < tol:
            if verbose: print(f'converge at iter={iter}, res={res} tol={tol}')
            break
    
     
    if verbose: 
        res = b - A(x)
        res = res.ravel() @ res.ravel()
        # if res > 1:
            # import ipdb; ipdb.set_trace()
        print(f'End at iter={iter}, res={res}')
    
    return x
