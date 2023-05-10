import torch
import numpy as np

# TODO: standardize the stopping criterion


def conjugate_gradient(A, b, x_init=None, tol=1e-6, num_iters=500, verbose=False):

    # Temp vars
    x = torch.zeros_like(b)
    r = torch.zeros_like(b)
    Ap = torch.zeros_like(b)

    # Initialize x
    if x_init is not None:
        x = x_init

    # Compute residual
    r = A(x)
    r *= -1.0
    r += b

    cg_tol = tol * torch.linalg.norm(b.ravel(), 2)  # Relative tol

    # CG iteration
    gamma_1 = p = None
    cg_iter = np.minimum(num_iters, np.prod(b.shape))
    for iter in range(cg_iter):

        # Check for convergence
        normr = torch.linalg.norm(r.ravel(), 2)
        if normr <= cg_tol:
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
