import torch
import cvxpy
import numpy as np
import scipy.optimize as sop
import matplotlib.pyplot as plt
import time
import random
from scipy import sparse, io
from lpsolvers import linprog_general, linprog_general_precond, linprog_general_precond_iter
# from lpsolvers_diff import LPProblem, LPSolverADMM, LPConvergenceLoss
from dprox.algo.lp import LPProblem, LPSolverADMM, LPConvergenceLoss
from utils import load_simple_cep_model, load_lp_componentes_from_mat
import copy
import pickle
from os.path import join
import logging


logname = './res_dprox.txt'

logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)


if torch.cuda.device_count() > 0:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    

def adjust_lr_cosine(optimizer, iteration, num_iters, base_lr, min_lr):
    ratio = iteration / num_iters
    lr = min_lr + (base_lr - min_lr) * (1.0 + np.cos(np.pi * ratio)) / 2.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
        
def generate_lp_problem(n, m1, m2, dtype=torch.float64, seed=0):
    n: int   # dimension of x
    m1: int  # number of equality constraints
    m2: int  # number of inequality constraints

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # c = torch.rand((n, 1), dtype=dtype, device=device) + 0.5  # create nonnegative price vector with mean 1
    c = torch.clip(torch.randn((n, 1), dtype=dtype, device=device), min=0.1)  # create nonnegative price vector 
    # c = torch.rand((n, 1), dtype=dtype, device=device)   # create nonnegative price vector 
    
    x01 = torch.clip(torch.randn((n//2, 1), dtype=dtype, device=device) * 1e2, min=0)  # create random solution vector
    x02 = torch.clip(torch.randn((n-(n//2), 1), dtype=dtype, device=device) * 1e-2, min=0)  # create random solution vector
    x0 = torch.vstack([x01, x02])
    
    A1 = torch.clip(torch.randn(m1//2, n, dtype=dtype, device=device) * 1e-1, min=0)  # create random, nonnegative matrix A
    A2 = torch.clip(torch.randn(m1-(m1//2), n, dtype=dtype, device=device) * 1e4, min=0)  # create random, nonnegative matrix A
    A = torch.vstack([A1, A2])
    
    G1 = torch.clip(torch.randn(m2//2, n, dtype=dtype, device=device) * 1e5, min=0)  # create random, nonnegative matrix A
    G2 = torch.clip(torch.randn(m2-(m2//2), n, dtype=dtype, device=device) * 1e-2, min=0)  # create random, nonnegative matrix A
    G = torch.vstack([G1, G2])
    
    # x0 = torch.abs(torch.randn(n, 1, dtype=dtype, device=device))
    # A = torch.abs(torch.randn(m1, n, dtype=dtype, device=device) * 1)  # create random, nonnegative matrix A
    # G = torch.abs(torch.randn(m2, n, dtype=dtype, device=device) * 1)  # create random, nonnegative matrix A
    b = A @ x0
    g = G @ x0 * 1.2
    
    lpproblem = LPProblem(c, G, g, A, b, norm_ord=float('inf'), sparse=False, device=device)
    return lpproblem


def test_lp_problem(n, m1, m2, seed=0, max_iters=10000, abstol=1e-4, reltol=1e-5, dtype=torch.float64):    
    lpproblem = generate_lp_problem(n, m1, m2, dtype, seed)
    c, A_ub, b_ub, A_eq, b_eq, x_lb, x_ub = lpproblem.unpack(to_numpy=True)
    lpadmm = LPSolverADMM(problem_scale=None, abstol=abstol, reltol=reltol, max_iters=max_iters, dtype=dtype, verbose=True).to(device)
    lpadmm.eval()
    print(lpadmm)
    
    start = time.time()
    with torch.no_grad():
        x, history, res = lpadmm.solve(lpproblem, rho=1e3, alpha=1.6, residual_balance=True, direct=False, polish=False)
    print(res[0])
    # print(torch.linalg.vector_norm(A @ x - b) / torch.linalg.vector_norm(b))
    x = x.cpu().numpy().squeeze()
    print(f'x >= 0 err: {x.min():.2e}')
    print(f'Ax = b err: {np.abs(b_eq - A_eq @ x).max():.2e}')
    print(f'Ax <= b err: {min(np.min(b_ub - A_ub @ x), 0):.2e}')
    
    torch.cuda.current_stream().synchronize()
    end = time.time()
    print(f"Time Elapse (LPADMM): {end-start:.4f}s\n")

    # plt.plot(history['objval'])
    # plt.plot(history['r_norm'], label='r_norm')
    # plt.plot(history['s_norm'], label='s_norm')
    # plt.plot(history['eps_primal'], label='eps_primal')
    # plt.plot(history['eps_dual'], label='eps_dual')
    # plt.legend()
    # plt.yscale('log')
    # plt.show()
    
    # start = time.time()
    # res = sop.linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), 
    #                   options={'maxiter': 10000, 'disp': True, 'presolve': True, 'autoscale': True})
    # # res = sop.linprog(c=c, A_eq=A, b_eq=b, bounds=(0, None), method='highs', options={'maxiter': 10000, 'disp': True, 'presolve': False})
    # print(res.fun, res.success, res.status)
    # # feasibility check
    # # print(res.x.min())
    # # print(res.con, np.abs(res.con).max(), np.abs(b_eq - A_eq @ res.x).max())
    # # print(res.slack, res.slack.min())
    # x = res.x
    # print(f'x >= 0 err: {x.min():.2e}')
    # print(f'Ax = b err: {np.abs(b_eq - A_eq @ x).max():.2e}')
    # print(f'Ax <= b err: {np.min(b_ub - A_ub @ x):.2e}')
    # end = time.time()
    # print(f"Time Elapse (SCIPY): {end-start:.4f}s\n")


def test_lp_general():
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    
    n = 10550   # dimension of x
    m1 = 2120  # number of equality constraints
    m2 = 3130  # number of inequality constraints
    
    dtype = torch.float64
    c = torch.clip(torch.randn((n, 1), dtype=dtype, device=device), min=0)  # create nonnegative price vector 
    # c = torch.rand((n, 1), dtype=dtype, device=device) + 0.5  # create nonnegative price vector with mean 1
    # c = torch.rand((n, 1), dtype=dtype, device=device)   # create nonnegative price vector 
    # print(c.min(), c.max(), c.mean())
    x01 = torch.clip(torch.randn((n//2, 1), dtype=dtype, device=device) * 1e2, min=0)  # create random solution vector
    x02 = torch.clip(torch.randn((n-(n//2), 1), dtype=dtype, device=device) * 1e-2, min=0)  # create random solution vector
    x0 = torch.vstack([x01, x02])
    # x0 = torch.clip(torch.randn((n, 1), dtype=dtype, device=device) * 10, min=0)  # create random solution vector

    A1 = torch.clip(torch.randn(m1//2, n, dtype=dtype, device=device) * 1e-1, min=0)  # create random, nonnegative matrix A
    A2 = torch.clip(torch.randn(m1-(m1//2), n, dtype=dtype, device=device) * 1e4, min=0)  # create random, nonnegative matrix A
    A = torch.vstack([A1, A2])
    G1 = torch.clip(torch.randn(m2//2, n, dtype=dtype, device=device) * 1e5, min=0)  # create random, nonnegative matrix A
    G2 = torch.clip(torch.randn(m2-(m2//2), n, dtype=dtype, device=device) * 1e-2, min=0)  # create random, nonnegative matrix A
    G = torch.vstack([G1, G2])
    # A = torch.clip(torch.randn(m1, n, dtype=dtype, device=device) * 1, min=0)  # create random, nonnegative matrix A
    # G = torch.clip(torch.randn(m2, n, dtype=dtype, device=device) * 1, min=0)  # create random, nonnegative matrix A
    b = A @ x0
    g = G @ x0 + 0.1
    
    max_iters = 100000
    abstol = 1e-4
    reltol = 1e-5
    
    # start = time.time()
    # x, history = linprog_general_precond(c, A_ub=G, b_ub=g, A_eq=A, b_eq=b, rho=1e-1, sigma=1e-6, alpha=1.6, 
    #                                         norm_ord=float('inf'), abstol=abstol, reltol=reltol,
    #                                         max_iters=max_iters, dtype=dtype, device=device, verbose=True)
    # print(torch.linalg.vector_norm(A @ x - b) / torch.linalg.vector_norm(b))
    # print(x.min())
    # torch.cuda.current_stream().synchronize()
    # end = time.time()
    # print(f"Time Elapse (Inverse Precond1 inf-norm): {end-start:.4f}s\n")
    
    
    # start = time.time()
    # norm_ord = 2
    # x, history = linprog_general_precond(c, A_ub=G, b_ub=g, A_eq=A, b_eq=b, rho=1e-1, sigma=1e-6, alpha=1.6, 
    #                                         norm_ord=norm_ord, abstol=abstol, reltol=1e-6,
    #                                         max_iters=max_iters, dtype=dtype, device=device, verbose=True)
    # print(torch.linalg.vector_norm(A @ x - b) / torch.linalg.vector_norm(b))
    # print(x.min())
    # torch.cuda.current_stream().synchronize()
    # end = time.time()
    # print(f"Time Elapse (Inverse Precond1 {norm_ord}-norm): {end-start:.4f}s\n")

    
    # start = time.time()
    # x, history = linprog_general_precond_iter(c, A_ub=G, b_ub=g, A_eq=A, b_eq=b, rho=1e-1, sigma=1e-6, alpha=1.6, 
    #                                           abstol=abstol, reltol=reltol,
    #                                           max_iters=max_iters, dtype=dtype, device=device)
    # print(torch.linalg.vector_norm(A @ x - b) / torch.linalg.vector_norm(b))
    # print(x.min())
    # torch.cuda.current_stream().synchronize()
    # end = time.time()
    # print(f"Time Elapse (CG): {end-start:.4f}s\n")

    lpproblem = LPProblem(c, G, g, A, b, norm_ord=float('inf'), sparse=False, device=device)
    # lpadmm = LPSolverADMM(problem_scale=lpproblem.problem_scale, abstol=abstol, reltol=reltol, max_iters=max_iters, dtype=dtype).to(device)
    lpadmm = LPSolverADMM(problem_scale=None, abstol=abstol, reltol=reltol, max_iters=max_iters, dtype=dtype).to(device)
    print(lpadmm)
    base_lr = 1e-1
    optimizer = torch.optim.Adam(lpadmm.parameters(), lr=base_lr)
    # optimizer = torch.optim.Adam([lpadmm.rho, lpadmm.alpha, lpadmm.gamma_c_mul_sqrt, lpadmm.gamma_b_mul_sqrt], lr=base_lr)
    criterion = LPConvergenceLoss()
    
    
    start = time.time()
    with torch.no_grad():
        x, history, res = lpadmm.solve(lpproblem)
    print(res[0])
    print(torch.linalg.vector_norm(A @ x - b) / torch.linalg.vector_norm(b))
    print(x.min())
    torch.cuda.current_stream().synchronize()
    end = time.time()
    print(f"Time Elapse (LPADMM): {end-start:.4f}s\n")
    
    start = time.time()
    loss_log = []
    num_iters = 1
    best_loss = float('inf')
    for k in range(num_iters):
        adjust_lr_cosine(optimizer, k, num_iters, base_lr=base_lr, min_lr=base_lr / 10)
        objval, r_norm, s_norm, eps_primal, eps_dual = lpadmm.unrolled_forward(lpproblem, max_iters=10)
        optimizer.zero_grad()
        # define loss
        loss = criterion(r_norm, s_norm, eps_primal, eps_dual)
        loss.backward()
        optimizer.step()

        loss_log.append(loss.item())
        # if best_loss > loss.item():
        #     best_loss = loss.item()
        #     best_state_dict = copy.deepcopy(lpadmm.state_dict())
        
        print(loss.item())
        print(lpadmm)

    # lpadmm.load_state_dict(best_state_dict)
    # plt.plot(loss_log)
    # plt.show()
    print(lpadmm)
    end1 = time.time()

    with torch.no_grad():
        # x, history, res = lpadmm.solve(lpproblem, rho=1e-1, sigma=1e-6, alpha=1.6)
        x, history, res = lpadmm.solve(lpproblem)
    print(res[0])
    print(torch.linalg.vector_norm(A @ x - b) / torch.linalg.vector_norm(b))
    print(x.min())
    torch.cuda.current_stream().synchronize()
    end = time.time()
    print(f"Time Elapse (LPADMM train stage): {end1-start:.4f}s\n")
    print(f"Time Elapse (LPADMM learned): {end-start:.4f}s\n")
    
    # plt.plot(history['objval'])
    # plt.plot(history['r_norm'], label='r_norm')
    # plt.plot(history['s_norm'], label='s_norm')
    # plt.plot(history['eps_primal'], label='eps_primal')
    # plt.plot(history['eps_dual'], label='eps_dual')
    # plt.legend()
    # plt.yscale('log')
    # plt.show()
    
    c = c.cpu().numpy()
    A = A.cpu().numpy()
    b = b.cpu().numpy()
    G = G.cpu().numpy()
    g = g.cpu().numpy()
    
    start = time.time()
    res = sop.linprog(c=c, A_ub=G, b_ub=g, A_eq=A, b_eq=b, bounds=(0, None), 
                      options={'maxiter': 10000, 'disp': True, 'presolve': True, 'autoscale': True})
    # res = sop.linprog(c=c, A_eq=A, b_eq=b, bounds=(0, None), method='highs', options={'maxiter': 10000, 'disp': True, 'presolve': False})
    print(res.fun, res.success, res.status)
    end = time.time()
    print(f"Time Elapse (SCIPY): {end-start:.4f}s\n")
    
    # x = cvxpy.Variable(shape=x0.shape)
    # p = cvxpy.Problem(cvxpy.Minimize(c.T @ x), [x >= 0, A @ x == b, G @ x <= g])
    # start = time.time()
    # p.solve(solver=cvxpy.GUROBI, verbose=True)
    # # p.solve(solver=cvxpy.OSQP, verbose=True)
    # end = time.time()
    # print(f"Time Elapse (CVXPY): {end-start:.4f}s\n")


def test_lp_general_sparse(seed=0, max_iters=int(2e6), abstol=1e-6, reltol=1e-12):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # logging.info('test simple_cep_model_20220916...\n')
    c, A_ub, A_eq, b_ub, b_eq = load_simple_cep_model() # rho=1e-2
    # A_ub, b_ub, A_eq, b_eq, c, x_lb, x_ub = load_lp_componentes_from_mat('./IESP_P1.mat')  # rho=1e-4
    # A_ub, b_ub, A_eq, b_eq, c, x_lb, x_ub = load_lp_componentes_from_mat('./IESP_P2.mat')
    
    dtype = torch.float64
    norm_ord = float('inf')

    lpproblem = LPProblem(c, A_ub, b_ub, A_eq, b_eq, x_lb=None, x_ub=None, norm_ord=norm_ord, dtype=dtype, sparse=True, device=device)
    # lpproblem = LPProblem(c, A_ub, b_ub, A_eq, b_eq, x_lb=x_lb, x_ub=x_ub, norm_ord=norm_ord, dtype=dtype, sparse=True, device=device)
    lpadmm = LPSolverADMM(rho=1e-2, problem_scale=None, abstol=abstol, reltol=reltol, max_iters=max_iters, dtype=dtype).to(device)
    
    optimizer = torch.optim.Adam(lpadmm.parameters(), lr=5e-3)
    criterion = LPConvergenceLoss()

    print(lpadmm)
    
    start = time.time()
    # loss_log = []
    # num_iters = 10
    # best_loss = float('inf')
    
    # for k in range(num_iters):
    #     # adjust_lr_cosine(optimizer, k, num_iters, base_lr=base_lr, min_lr=1e-3)
    #     optimizer.zero_grad()
    #     _, _, res = lpadmm.solve(lpproblem, max_iters=100)
    #     objval, r_norm, s_norm, eps_primal, eps_dual = res
        
    #     # define loss
    #     loss = criterion(r_norm, s_norm, eps_primal, eps_dual)
    #     loss.backward()
    #     optimizer.step()

    #     loss_log.append(loss.item())
    #     print(loss.item())
    #     print(lpadmm)
    
    # print(lpadmm)
    end1 = time.time()

    with torch.no_grad():
        lpadmm.eval()
        x, history, res = lpadmm.solve(lpproblem, residual_balance=True)
    
    print(res[0])
    # print(torch.linalg.vector_norm(lpproblem.A_eq @ x - lpproblem.b_eq) / torch.linalg.vector_norm(lpproblem.b_eq))
    # print(x.min())
    x = x.cpu().numpy().squeeze()
    print(f'x >= 0 err: {x.min():.2e}')
    print(f'Ax = b err: {np.abs(b_eq - A_eq @ x).max():.2e}')
    print(f'Ax <= b err: {min(np.min(b_ub - A_ub @ x), 0):.2e}')

    torch.cuda.current_stream().synchronize()
    end = time.time()
    logging.info(f"Time Elapse (DProx train stage): {end1-start:.4f}s\n")
    logging.info(f"Time Elapse (DProx ft): {end-start:.4f}s\n")
    
    
if __name__ == '__main__':
    # test_lp_problem(n=2000, m1=2000, m2=2000, seed=0, max_iters=int(2e6), abstol=1e-6, reltol=1e-12)
    # test_lp_general()
    test_lp_general_sparse(seed=0, max_iters=int(1e5), abstol=1e-2, reltol=1e-10)
