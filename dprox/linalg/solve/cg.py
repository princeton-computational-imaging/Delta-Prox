import torch
import numpy as np

# TODO: standardize the stopping criterion
RTOL = 1e-6
MAX_ITERS = 100


def conjugate_gradient(
    A, b, 
    x0=None, rtol=RTOL, max_iters=MAX_ITERS, verbose=False
):

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

    cg_tol = rtol * torch.linalg.norm(b.ravel(), 2)  # Relative tolerence

    # CG iteration
    gamma_1 = p = None
    cg_iter = np.minimum(max_iters, np.prod(b.shape))
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

    if verbose:
        print(f'Not converged, r norm={normr.item()}')
    
    return x


def conjugate_gradient2(
    A, b, 
    x0=None, rtol=RTOL, max_iters=500, verbose=False
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


def PCG(
    A, b, 
    Minv=None, x0=None, rtol=RTOL, max_iters=MAX_ITERS, verbose=False
):
    """Preconditioned Conjugate Gradient Method"""
    if Minv is None:
        Minv = lambda x: x

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


def cg_batch(A_bmm, B, M_bmm=None, X0=None, rtol=RTOL, atol=0., maxiter=None, verbose=False):
    """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.
    This function solves a batch of matrix linear systems of the form
        A_i X_i = B_i,  i=1,...,K,
    where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
    and X_i is the n x m matrix representing the solution for the ith system.
    Args:
        A_bmm: A callable that performs a batch matrix multiply of A and a K x n x m matrix.
        B: A K x n x m matrix representing the right hand sides.
        M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
            matrices M and a K x n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """
    K, n, m = B.shape

    if M_bmm is None:
        M_bmm = lambda x: x
    if X0 is None:
        X0 = M_bmm(B)
    if maxiter is None:
        maxiter = 5 * n

    assert B.shape == (K, n, m)
    assert X0.shape == (K, n, m)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    X_k = X0
    R_k = B - A_bmm(X_k)
    Z_k = M_bmm(R_k)

    P_k = torch.zeros_like(Z_k)

    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k

    B_norm = torch.norm(B, dim=1)
    stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

    if verbose:
        print("%03s | %010s %06s" % ("it", "dist", "it/s"))

    optimal = False
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        start_iter = time.perf_counter()
        Z_k = M_bmm(R_k)

        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2 * Z_k2).sum(1)
            denominator[denominator == 0] = 1e-8
            beta = (R_k1 * Z_k1).sum(1) / denominator
            P_k = Z_k1 + beta.unsqueeze(1) * P_k1

        denominator = (P_k * A_bmm(P_k)).sum(1)
        denominator[denominator == 0] = 1e-8
        alpha = (R_k1 * Z_k1).sum(1) / denominator
        X_k = X_k1 + alpha.unsqueeze(1) * P_k
        R_k = R_k1 - alpha.unsqueeze(1) * A_bmm(P_k)
        end_iter = time.perf_counter()

        residual_norm = torch.norm(A_bmm(X_k) - B, dim=1)

        # if verbose:
        #     print("%03d | %8.4e %4.2f" %
        #           (k, torch.max(residual_norm-stopping_matrix),
        #             1. / (end_iter - start_iter)))

        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break

    end = time.perf_counter()

    if verbose:
        if optimal:
            print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                  (k, (end - start) * MAX_ITERS0))
        else:
            print("Terminated in %d steps (optimal). Took %.3f ms." %
                  (k, (end - start) * MAX_ITERS0))


    info = {
        "niter": k,
        "optimal": optimal
    }

    return X_k, info