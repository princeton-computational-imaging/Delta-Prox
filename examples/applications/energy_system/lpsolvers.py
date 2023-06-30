import torch
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from linear_solvers import PLSS, PCG
from utils import *


def linprog_eq(c, A, b, rho, alpha, device=None):
    # handle equality constraint Ax == b and x >= 0
    MAX_ITER = 5000
    ABSTOL = 1e-4
    RELTOL = 1e-3

    m, n = A.shape

    x = torch.zeros(n, 1, device=device)
    z = torch.zeros(n, 1, device=device)
    u = torch.zeros(n, 1, device=device)
    
    left = torch.vstack([
        torch.hstack([rho * torch.eye(n, device=device), A.T]),
        torch.hstack([A, torch.zeros((m, m), device=device)])
    ])
    # A_fn = lambda x: left @ x
    tmp = None
    inverse_left = torch.inverse(left)
    
    history = defaultdict(lambda : [])
    for k in tqdm(range(MAX_ITER)):
        # x-update
        right = torch.vstack([rho * (z - u) - c, b])
        tmp = inverse_left @ right
        # tmp = cg(A_fn, right, None, tol=1e-6, num_iters=500)
                        
        x = tmp[:n]

        # z-update with relaxation
        zold = z
        x_hat = alpha * x + (1 - alpha) * zold
        z = torch.clip(x_hat + u, min=0)
        u = u + (x_hat - z)

        # diagnostics, reporting, termination checks
        objval = c.T @ x
        r_norm = torch.linalg.vector_norm(x - z)
        s_norm = torch.linalg.vector_norm(-rho * (z - zold))
        eps_primal = n**0.5 * ABSTOL + RELTOL * max(torch.linalg.vector_norm(x), torch.linalg.vector_norm(-z))
        eps_dual = n**0.5 * ABSTOL + RELTOL * torch.linalg.vector_norm(rho * u)
        history['r_norm'].append(r_norm.item())
        history['s_norm'].append(s_norm.item())
        history['eps_primal'].append(eps_primal.item())
        history['eps_dual'].append(eps_dual.item())
        history['objval'].append(objval.item())
        
        if r_norm < eps_primal and s_norm < eps_dual:
            break
        
    print(objval.item())
    return x, history


# QSQP formulation 
def linprog_general(c, A_ub, b_ub, A_eq, b_eq, rho, sigma, alpha, max_iters=5000, dtype=torch.float64, device=None, verbose=False):
    MAX_ITER = max_iters
    ABSTOL = 1e-4
    RELTOL = 1e-3
    
    norm_ord = float('inf')
    m_ub = A_ub.shape[0]
    m_eq = A_eq.shape[0]
    n = c.shape[0]
    
    A = torch.vstack([A_ub, A_eq, torch.eye(n, dtype=dtype, device=device)])
    m, _ = A.shape
        
    # rho_vec = torch.ones(m, 1, dtype=dtype, device=device) * rho
    # rho_vec[m_ub:m_ub+m_eq] *= 1e3
    # inv_rho_vec = 1 / rho_vec
    
    left = torch.vstack([
        torch.hstack([sigma * torch.eye(n, dtype=dtype, device=device), A.T]),
        torch.hstack([A, -1/rho * torch.eye(m, dtype=dtype, device=device)])
    ])
    
    inverse_left = torch.inverse(left)

    x = torch.zeros(n, 1, dtype=dtype, device=device)
    z = torch.zeros(m, 1, dtype=dtype, device=device)
    y = torch.zeros(m, 1, dtype=dtype, device=device)
    xtilde = torch.zeros_like(x)
    ztilde = torch.zeros_like(z)

    lb = torch.vstack([-float('inf') * torch.ones(m_ub, 1, device=device), b_eq, torch.zeros(n, 1, device=device)])
    ub = torch.vstack([b_ub, b_eq, float('inf') * torch.ones(n, 1, dtype=dtype, device=device)])
    
    def proj(z):
        z = torch.clip(z, min=lb, max=ub)
        return z
    
    c_norm = torch.linalg.vector_norm(c, ord=norm_ord)
    history = defaultdict(lambda : [])
    for k in tqdm(range(MAX_ITER)):
        # x-update
        right = torch.vstack([sigma * x - c, z - 1/rho * y])
        # right = torch.vstack([sigma * x - c, z - inv_rho_vec * y])
        tmp = inverse_left @ right
        xtilde = tmp[:n]
        v = tmp[n:]
        ztilde = z + (1/rho) * (v - y)
        # ztilde = z + inv_rho_vec * (v - y)
        x = alpha * xtilde + (1 - alpha) * x

        # z-update with relaxation
        z_hat = alpha * ztilde + (1 - alpha) * z
        z = proj(z_hat + 1/rho * y)
        # z = proj(z_hat + inv_rho_vec * y)
        
        y = y + rho * (z_hat - z)
        # y = y + rho_vec * (z_hat - z)

        # diagnostics, reporting, termination checks
        if k % 50 == 0:
            objval = c.T @ x
            Ax = A @ x
            ATy = A.T @ y
            r_norm = torch.linalg.vector_norm(Ax - z, ord=norm_ord)
            s_norm = torch.linalg.vector_norm(c + ATy, ord=norm_ord)
            eps_primal = m**0.5 * ABSTOL + RELTOL * max(torch.linalg.vector_norm(Ax, ord=norm_ord), torch.linalg.vector_norm(z, ord=norm_ord))
            eps_dual = n**0.5 * ABSTOL + RELTOL * max(torch.linalg.vector_norm(ATy, ord=norm_ord), c_norm)
            history['r_norm'].append(r_norm.item())
            history['s_norm'].append(s_norm.item())
            history['eps_primal'].append(eps_primal.item())
            history['eps_dual'].append(eps_dual.item())
            history['objval'].append(objval.item())
            
            if verbose:
                if k % 1000 == 0:
                    print(objval.item(), r_norm.item(), s_norm.item(), eps_primal.item(), eps_dual.item(), float(rho))                        
            
            if r_norm < eps_primal and s_norm < eps_dual:
                print(objval.item(), r_norm.item(), s_norm.item(), eps_primal.item(), eps_dual.item(), float(rho))                        
                break

    print(objval.item())
    return x, history


@torch.no_grad()
def linprog_general_precond(c, A_ub, b_ub, A_eq, b_eq, rho, sigma, alpha, abstol=1e-4, reltol=1e-3,
                               norm_ord=float('inf'), max_iters=5000, dtype=torch.float64, device=None, verbose=False):
    # NOTE using float64 is critical otherwise the algorithm cannot converge to a high numerical precision 
    MAX_ITER = max_iters
    ABSTOL = abstol
    RELTOL = reltol
    
    # vector_norm = partial(torch.linalg.vector_norm, ord=float('inf'))
    vector_norm = partial(torch.linalg.vector_norm, ord=norm_ord)
    m_ub = A_ub.shape[0]
    m_eq = A_eq.shape[0]
    n = c.shape[0]
    
    A = torch.vstack([A_ub, A_eq, torch.eye(n, device=device)])
    m, _ = A.shape
    lb = torch.vstack([-float('inf') * torch.ones(m_ub, 1, device=device), b_eq, torch.zeros(n, 1, device=device)])
    # ub = torch.vstack([b_ub, b_eq, float('inf') * torch.ones(n, 1, device=device)])
    b = torch.vstack([b_ub, b_eq])
    ub = torch.vstack([b, float('inf') * torch.ones(n, 1, device=device)])
    
    d, e, gamma_c, gamma_b, A = Ruiz_equilibration_th(A, c, b, ord=norm_ord, max_iters=100)
    
    d = d.view(n, 1)
    e = e.view(m, 1)
    
    c_ori = c
    c = gamma_c * (d * c)
    
    # rho = rho * gamma_c
    lb = e * lb * gamma_b
    ub = e * ub * gamma_b
    m, _ = A.shape
    
    left = torch.vstack([
        torch.hstack([sigma * torch.eye(n, device=device), A.T]),
        torch.hstack([A, -1/rho * torch.eye(m, device=device)])
    ])
    
    inverse_left = torch.inverse(left)

    x = torch.zeros(n, 1, dtype=dtype, device=device)
    z = torch.zeros(m, 1, dtype=dtype, device=device)
    y = torch.zeros(m, 1, dtype=dtype, device=device)
    xtilde = torch.zeros_like(x)
    ztilde = torch.zeros_like(z)

    def proj(z):
        z = torch.clip(z, min=lb, max=ub)
        return z
        
    history = defaultdict(lambda : [])
    for k in tqdm(range(MAX_ITER)):
        # x-update
        right = torch.vstack([sigma * x - c, z - 1/rho * y])
        tmp = inverse_left @ right
        xtilde = tmp[:n]
        v = tmp[n:]
        ztilde = z + (1/rho) * (v - y)
        x = alpha * xtilde + (1 - alpha) * x

        # z-update with relaxation
        z_hat = alpha * ztilde + (1 - alpha) * z
        z = proj(z_hat + 1/rho * y)
        
        y = y + rho * (z_hat - z)

        # diagnostics, reporting, termination checks
        if k % 25 == 0:            
            objval = c_ori.T @ (x * d / gamma_b)
            Ax = A @ x
            ATy = A.T @ y
            r_norm = vector_norm((Ax - z) / e / gamma_b)
            s_norm = vector_norm((c + ATy) / d / gamma_c)
            eps_primal = m**0.5 * ABSTOL + RELTOL * max(vector_norm(Ax / e / gamma_b), vector_norm(z / e / gamma_b))
            eps_dual = n**0.5 * ABSTOL + RELTOL * max(vector_norm(ATy / d / gamma_c),  vector_norm(c / d / gamma_c))
            
            history['r_norm'].append(r_norm.item())
            history['s_norm'].append(s_norm.item())
            history['eps_primal'].append(eps_primal.item())
            history['eps_dual'].append(eps_dual.item())
            history['objval'].append(objval.item())

            if verbose:
                if k % 1000 == 0:
                    print(objval.item(), r_norm.item(), s_norm.item(), eps_primal.item(), eps_dual.item(), float(rho))            
            
            if r_norm < eps_primal and s_norm < eps_dual:
                print(objval.item(), r_norm.item(), s_norm.item(), eps_primal.item(), eps_dual.item(), float(rho))            
                break
        
    print(objval.item())
    return x * d / gamma_b, history


# use iterative linear solver
@torch.no_grad()
def linprog_general_precond_iter(c, A_ub, b_ub, A_eq, b_eq, rho, sigma, alpha, abstol=1e-4, reltol=1e-4,
                                 norm_ord=float('inf'), max_iters=5000, dtype=torch.float64, device=None, verbose=True):
    # NOTE using float64 is critical otherwise the algorithm cannot converge to a high numerical precision 
    MAX_ITER = max_iters
    ABSTOL = abstol
    RELTOL = reltol
    
    vector_norm = partial(torch.linalg.vector_norm, ord=norm_ord)
    m_ub = A_ub.shape[0]
    m_eq = A_eq.shape[0]
    n = c.shape[0]
    
    A = torch.vstack([A_ub, A_eq, torch.eye(n, dtype=dtype, device=device)])
    m, _ = A.shape
    lb = torch.vstack([-float('inf') * torch.ones(m_ub, 1, device=device), b_eq, torch.zeros(n, 1, device=device)])
    b = torch.vstack([b_ub, b_eq])
    ub = torch.vstack([b, float('inf') * torch.ones(n, 1, device=device)])
    
    d, e, gamma_c, gamma_b, A = Ruiz_equilibration_th(A, c, b, ord=norm_ord, max_iters=100)
    
    d = d.view(n, 1)
    e = e.view(m, 1)
    
    c_ori = c
    c = gamma_c * (d * c)
    lb = e * lb * gamma_b
    ub = e * ub * gamma_b
    m, _ = A.shape

    x = torch.zeros(n, 1, dtype=dtype, device=device)
    z = torch.zeros(m, 1, dtype=dtype, device=device)
    y = torch.zeros(m, 1, dtype=dtype, device=device)
    xtilde = torch.zeros_like(x)
    ztilde = torch.zeros_like(z)

    ATA = sigma * torch.eye(n, dtype=dtype, device=device) + rho * (A.T @ A)
    ATAfun = lambda x: ATA @ x
    Aop = LinearOp(A_fun=ATAfun, AT_fun=ATAfun, shape=(n, n))
    
    M = sigma * torch.ones(n, dtype=dtype, device=device) + rho * (torch.linalg.norm(A, dim=0) ** 2)
    M = M.unsqueeze(1)
    Minv = lambda x: x / M
    
    btols = torch.logspace(-6, -10, 10000)
    history = defaultdict(lambda : [])
    for k in tqdm(range(MAX_ITER)):
        # x-update
        right = sigma * x - c + A.T @ (rho * z - y)
        
        btol = 1e-10 if k >= 10000 else btols[k]        
        xtilde = PCG(Aop, right, Minv, x_init=xtilde, btol=btol, num_iters=500, verbose=False)
        ztilde = A @ xtilde
        x = alpha * xtilde + (1 - alpha) * x

        # z-update with relaxation
        z_hat = alpha * ztilde + (1 - alpha) * z
        z = torch.clip(z_hat + 1/rho * y, min=lb, max=ub)
        
        y = y + rho * (z_hat - z)

        # diagnostics, reporting, termination checks
        if k % 25 == 0:
            objval = c_ori.T @ (x * d / gamma_b)
            Ax = A @ x
            ATy = A.T @ y
            r_norm = vector_norm((Ax - z) / e / gamma_b)
            s_norm = vector_norm((c + ATy) / d / gamma_c)
            eps_primal = m**0.5 * ABSTOL + RELTOL * max(vector_norm(Ax / e / gamma_b), vector_norm(z / e / gamma_b))
            eps_dual = n**0.5 * ABSTOL + RELTOL * max(vector_norm(ATy / d / gamma_c),  vector_norm(c / d / gamma_c))

            # update rho (not work well)
            # coeff = (vector_norm(Ax - z) / max(vector_norm(Ax), vector_norm(z))) / (vector_norm(c + ATy) / max(vector_norm(ATy), vector_norm(c)))
            # rho = rho * (coeff ** 0.5)

            # ATA = sigma * torch.eye(n, dtype=dtype, device=device) + rho * (A.T @ A)
            # ATAfun = lambda x: ATA @ x
            # Aop = LinearOp(A_fun=ATAfun, AT_fun=ATAfun, shape=(n, n))
            
            # update btol
            # btol = max(0.15 * torch.sqrt(vector_norm(Ax - z, ord=2) * vector_norm(c + ATy, ord=2)), btol_min)
            # print(btol)
            
            history['r_norm'].append(r_norm.item())
            history['s_norm'].append(s_norm.item())
            history['eps_primal'].append(eps_primal.item())
            history['eps_dual'].append(eps_dual.item())
            history['objval'].append(objval.item())
            history['rho'].append(rho)
            
            if verbose:
                if k % 1000 == 0:
                    print(objval.item(), r_norm.item(), s_norm.item(), eps_primal.item(), eps_dual.item(), float(rho))            
            
            if r_norm < eps_primal and s_norm < eps_dual:
                print(objval.item(), r_norm.item(), s_norm.item(), eps_primal.item(), eps_dual.item(), float(rho))            
                break
        
    print(objval.item())
    return x * d / gamma_b, history


# A is preconditioned in numpy
def linprog_general_precond_sparse(
    c, A, AT, b_ub, b_eq, ruiz_norms, Minv, rho, sigma, alpha, 
    max_iters=5000, abstol=1e-4, reltol=1e-4, dtype=torch.float64, device=None, verbose=False):
    
    MAX_ITER = max_iters
    ABSTOL = abstol
    RELTOL = reltol
    
    norm_ord = float('inf')
    # norm_ord = 2
    vector_norm = partial(torch.linalg.vector_norm, ord=norm_ord)
    m_ub = b_ub.shape[0]
    m_eq = b_eq.shape[0]
    n = c.shape[0]
    m, _ = A.shape
    
    b_ub = b_ub.view(m_ub, 1)
    b_eq = b_eq.view(m_eq, 1)
    c = c.view(n, 1)
    d, e, gamma_c, gamma_b = ruiz_norms
    d = d.view(n, 1)
    e = e.view(m, 1)
    
    lb = torch.vstack([-float('inf') * torch.ones(m_ub, 1, device=device), b_eq, torch.zeros(n, 1, device=device)])
    ub = torch.vstack([b_ub, b_eq, float('inf') * torch.ones(n, 1, device=device)])

    c_ori = c
    c = gamma_c * (d * c)
    # rho = rho * gamma_c
    # sigma = sigma * gamma_c
    lb = e * lb * gamma_b
    ub = e * ub * gamma_b

    x = torch.zeros(n, 1, dtype=dtype, device=device)
    z = torch.zeros(m, 1, dtype=dtype, device=device)
    y = torch.zeros(m, 1, dtype=dtype, device=device)
    xtilde = torch.zeros_like(x)
    ztilde = torch.zeros_like(z)
    
    ATAfun = LPATA_Func(A, AT, rho, sigma)
    Aop = LinearOp(A_fun=ATAfun, AT_fun=ATAfun, shape=(n, n))
    
    btols = torch.logspace(-6, -10, 10000)

    history = defaultdict(lambda : [])
    for k in tqdm(range(MAX_ITER)):
        # x-update
        right = sigma * x - c + AT @ (rho * z - y)

        btol = 1e-10 if k >= 10000 else btols[k]
        xtilde = PCG(Aop, right, Minv, x_init=xtilde, btol=btol, num_iters=500, verbose=False)
        # xtilde, _ = PLSS(xtilde, Aop, right, atol=1e-6, btol=btols[k], maxiter=n)
        ztilde = A @ xtilde
        x = alpha * xtilde + (1 - alpha) * x

        # z-update with relaxation
        z_hat = alpha * ztilde + (1 - alpha) * z
        z = torch.clip(z_hat + 1/rho * y, min=lb, max=ub)
        
        y = y + rho * (z_hat - z)

        # diagnostics, reporting, termination checks
        if k % 25 == 0:
            objval = c_ori.T @ (x * d / gamma_b)
            Ax = A @ x
            ATy = AT @ y
            r_norm = vector_norm((Ax - z) / e / gamma_b)
            s_norm = vector_norm((c + ATy) / d / gamma_c)
            eps_primal = m**0.5 * ABSTOL + RELTOL * max(vector_norm(Ax / e / gamma_b), vector_norm(z / e / gamma_b))
            eps_dual = n**0.5 * ABSTOL + RELTOL * max(vector_norm(ATy / d / gamma_c),  vector_norm(c / d / gamma_c))
            
            # update rho
            # rho = rho * torch.sqrt((r_norm / eps_primal) / (s_norm / eps_dual))
            # ATAfun = LPATA_Func(A, AT, rho, sigma)
            # Aop = LinearOp(A_fun=ATAfun, AT_fun=ATAfun, shape=(n, n))
            
            history['r_norm'].append(r_norm.item())
            history['s_norm'].append(s_norm.item())
            history['eps_primal'].append(eps_primal.item())
            history['eps_dual'].append(eps_dual.item())
            history['objval'].append(objval.item())
            
            if verbose:
                if k % 1000 == 0:
                    print(objval.item(), r_norm.item(), s_norm.item(), eps_primal.item(), eps_dual.item(), float(rho))

                    # r_norm_bar = vector_norm((Ax - z))
                    # s_norm_bar = vector_norm((c + ATy))
                    # eps_primal_bar = m**0.5 * ABSTOL + RELTOL * max(vector_norm(Ax), vector_norm(z))
                    # eps_dual_bar = n**0.5 * ABSTOL + RELTOL * max(vector_norm(ATy),  vector_norm(c))
                    # print(r_norm_bar.item(), s_norm_bar.item(), eps_primal_bar.item(), eps_dual_bar.item())
                                
            if r_norm < eps_primal and s_norm < eps_dual:
                print(objval.item(), r_norm.item(), s_norm.item(), eps_primal.item(), eps_dual.item(), float(rho))
                break
    
    print(objval.item())
    return x * d / gamma_b, history
