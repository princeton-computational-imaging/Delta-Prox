from collections import defaultdict
from functools import partial

import scipy
import torch
import torch.nn as nn
from tqdm import tqdm

from dprox.linalg.custom import pcg
from .utils import *


class LPConvergenceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, r_norm, s_norm, eps_primal, eps_dual):
        primal_loss = (r_norm / eps_primal)
        dual_loss = (s_norm / eps_dual)
        # balancing_loss = torch.abs(primal_loss - dual_loss)
        # loss = primal_loss + dual_loss + balancing_loss
        loss = torch.log(primal_loss) + torch.log(dual_loss)
        return loss


class LPProblem:
    def __init__(self, c, A_ub, b_ub, A_eq, b_eq, x_lb=None, x_ub=None, norm_ord=float('inf'), dtype=torch.float64, sparse=False, device=None) -> None:
        self.c = c
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.A_eq = A_eq
        self.b_eq = b_eq
        n = c.shape[0]
        self.device = device
        if x_lb is None:
            # x_lb = torch.zeros(n, 1, dtype=dtype, device=device)
            x_lb = np.zeros((n, 1))
        if x_ub is None:
            # x_ub = float('inf') * torch.ones(n, 1, dtype=dtype, device=device)
            x_ub = float('inf') * np.ones((n, 1))
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.original_problem = self.c, self.A_ub, self.b_ub, self.A_eq, self.b_eq, self.x_lb, self.x_ub
        self._preprocess(norm_ord, sparse)

    def unpack(self):
        return self.original_problem

    @torch.no_grad()
    def _preprocess(self, norm_ord, sparse):
        c, A_ub, b_ub, A_eq, b_eq, x_lb, x_ub = self.original_problem
        device = self.device
        m_ub = A_ub.shape[0]
        m_eq = A_eq.shape[0]
        n = c.shape[0]
        if not sparse:
            A = torch.vstack([A_ub, A_eq, torch.eye(n, device=device)])
            self.d, self.e, self.gamma_c, self.gamma_b, self.A = Ruiz_equilibration_th(A, c, b=torch.concatenate([b_ub, b_eq, x_ub]), ord=norm_ord, max_iters=100)
            self.AT = self.A.T
            self.Acnorm = torch.linalg.norm(A, axis=0)
        else:  # NOTE use numpy instead as many functions are not supported in torch.sparse
            A = scipy.sparse.vstack([A_ub, A_eq, scipy.sparse.eye(n)])
            m, _ = A.shape
            d, e, gamma_c, gamma_b, A = Ruiz_equilibration_sparse_np(A, c, b=np.concatenate([b_ub, b_eq, x_ub[:, 0]]), ord=float('inf'), max_iters=20, verbose=True)
            Acnorm = slinalg.norm(A, axis=0)
            self.A, self.AT = scipy_sparse_to_torchop(A, device=device)
            self.A_eq = scipy_sparse_to_torchop(A_eq, device=device, output_AT=False)
            self.A_ub = scipy_sparse_to_torchop(A_ub, device=device, output_AT=False)
            self.c = c = torch.from_numpy(c).to(device).view(n, 1)
            self.b_eq = b_eq = torch.from_numpy(b_eq).to(device).view(m_eq, 1)
            self.b_ub = b_ub = torch.from_numpy(b_ub).to(device).view(m_ub, 1)
            self.d = torch.from_numpy(d).to(device).view(1, n)
            self.e = torch.from_numpy(e).to(device).view(m, 1)
            self.Acnorm = torch.from_numpy(Acnorm).to(device)
            self.gamma_b = gamma_b
            self.gamma_c = gamma_c

        if isinstance(x_lb, np.ndarray):
            self.x_lb = x_lb = torch.from_numpy(x_lb).to(device).view(n, 1)
        if isinstance(x_ub, np.ndarray):
            self.x_ub = x_ub = torch.from_numpy(x_ub).to(device).view(n, 1)

        lb = torch.vstack([-float('inf') * torch.ones(m_ub, 1, device=device), b_eq, x_lb])
        ub = torch.vstack([b_ub, b_eq, x_ub])
        self.lb, self.ub = lb, ub

    def get_data(self):
        return self.d.detach().clone(), self.e.detach().clone(),\
            self.gamma_c, self.gamma_b, \
            self.A, self.AT, self.Acnorm, \
            self.ub.detach().clone(), self.lb.detach().clone(), self.c.detach().clone()

    @property
    def problem_scale(self):
        return self.A.shape


class LPSolverADMM(nn.Module):
    def __init__(self, rho=1, problem_scale=None, abstol=1e-4, reltol=1e-3, norm_ord=float('inf'), max_iters=5000, dtype=torch.float64, verbose=True) -> None:
        super().__init__()
        self.problem_scale = problem_scale
        self.abstol = abstol
        self.reltol = reltol
        self.norm_ord = norm_ord
        self.max_iters = max_iters
        self.dtype = dtype
        self.verbose = verbose
        self._init_params(rho=rho)

    def _init_params(self, rho):
        self.rho = nn.Parameter(torch.tensor(rho, dtype=self.dtype))
        # self.rho_log = nn.Parameter(torch.tensor(np.log(rho), dtype=self.dtype))

        # self.sigma = nn.Parameter(torch.tensor(1e-6, dtype=self.dtype))
        self.sigma_log = nn.Parameter(torch.tensor(np.log(1e-6), dtype=self.dtype))
        self.alpha = nn.Parameter(torch.tensor(1.6, dtype=self.dtype))

        self.gamma_c_mul = nn.Parameter(torch.tensor(1., dtype=self.dtype))
        self.gamma_b_mul = nn.Parameter(torch.tensor(1., dtype=self.dtype))

        # self.gamma_c_mul_log = nn.Parameter(torch.tensor(0., dtype=self.dtype))
        # self.gamma_b_mul_log = nn.Parameter(torch.tensor(0., dtype=self.dtype))

        # self.register_buffer('sigma', torch.tensor(1e-6, dtype=self.dtype))
        # self.register_buffer('alpha', torch.tensor(1.6, dtype=self.dtype))

        if self.problem_scale is not None:
            m, n = self.problem_scale
            self.d_mul_log = nn.Parameter(torch.zeros(1, n, dtype=self.dtype))
            self.e_mul_log = nn.Parameter(torch.zeros(m, 1, dtype=self.dtype))

    def solve(self, lpproblem, rho=None, sigma=None, alpha=None, max_iters=None, eval_freq=25, residual_balance=False, direct=False, polish=False):
        if max_iters is None: max_iters = self.max_iters
        vector_norm = partial(torch.linalg.vector_norm, ord=self.norm_ord)
        d, e, gamma_c, gamma_b, A, AT, Acnorm, ub, lb, c = lpproblem.get_data()
        m, n = A.shape
        m_ub = lpproblem.A_ub.shape[0]

        device = c.device
        dtype = self.dtype

        rho = rho if rho is not None else self.rho
        # rho = rho if rho is not None else torch.exp(self.rho_log)
        sigma = sigma if sigma is not None else torch.exp(self.sigma_log)
        # sigma = sigma if sigma is not None else self.sigma
        alpha = alpha if alpha is not None else self.alpha

        if self.problem_scale is not None:  # NOTE do not use as it doesn't work well
            d_mul = torch.exp(self.d_mul_log)
            e_mul = torch.exp(self.e_mul_log)
            d = d * d_mul
            e = e * e_mul
            A = A * e_mul * d_mul

        # gamma_c = torch.exp(self.gamma_c_mul_log) * gamma_c
        # gamma_b = torch.exp(self.gamma_b_mul_log) * gamma_b

        gamma_c = self.gamma_c_mul * gamma_c
        gamma_b = self.gamma_b_mul * gamma_b
        d = d.view(n, 1)
        e = e.view(m, 1)
        c = gamma_c * (d * c)

        mask_lb = ~torch.isinf(lb)
        mask_ub = ~torch.isinf(ub)
        lb[mask_lb] *= gamma_b * e[mask_lb]
        ub[mask_ub] *= gamma_b * e[mask_ub]

        # rho_base = torch.ones(m, 1, dtype=dtype, device=device)
        # mask_rho = (lb == ub)
        # rho_base[mask_rho] *= 1e3
        # rho = rho_base * rho

        x = torch.zeros(n, 1, dtype=dtype, device=device)
        z = torch.zeros(m, 1, dtype=dtype, device=device)
        y = torch.zeros(m, 1, dtype=dtype, device=device)
        xtilde = torch.zeros_like(x)

        rtols = torch.logspace(-6, -10, 10000)
        history = defaultdict(lambda: [])

        Kinv = None
        if direct:
            K = (AT @ A) * rho + sigma * torch.eye(n, device=device, dtype=dtype)
            Kinv = torch.inverse(K)

        ATAfun = LPATA_Func(A, AT, rho, sigma)
        ATAop = LinearOp(A_fun=ATAfun, AT_fun=ATAfun, shape=(n, n))

        M_constant = sigma * torch.ones((n, 1), dtype=dtype, device=device)
        M = (M_constant + rho * (Acnorm.unsqueeze(1) ** 2))
        def Minv(x): return x / M

        for k in tqdm(range(max_iters)):
            rtol = 1e-10 if k >= 10000 else rtols[k]
            variables = x, z, y, xtilde
            x, z, y, xtilde = self._solve_one_iter_precond(variables, c, A, AT, ATAop, Minv, lb, ub, rtol, rho, sigma, alpha, Kinv=Kinv)

            if k % eval_freq == 0:
                objval, r_norm, s_norm, eps_primal, eps_dual = self.eval_result(c, A, AT, d, e, gamma_c, gamma_b, x, z, y)

                # update rho (residual balance)
                if residual_balance and k % 1000 == 0 and k != 0:
                    if r_norm > 10 * eps_primal or eps_dual > 10 * s_norm:
                        rho = rho * 2
                        flag = True
                    elif s_norm > 10 * eps_dual or eps_primal > 10 * r_norm:
                        rho = rho / 2
                        flag = True
                    else:
                        flag = False
                    if flag:
                        if direct:
                            K = (AT @ A) * rho + sigma * torch.eye(n, device=device, dtype=dtype)
                            Kinv = torch.inverse(K)
                        else:
                            M = (M_constant + rho * (Acnorm.unsqueeze(1) ** 2))
                            def Minv(x): return x / M
                            ATAfun = LPATA_Func(A, AT, rho, sigma)
                            ATAop = LinearOp(A_fun=ATAfun, AT_fun=ATAfun, shape=(n, n))

                # if residual_balance and k >= 10000 and k % 500 == 0:
                #     if (r_norm / eps_primal) > (1 / self.reltol / 1e2) * (s_norm / eps_dual):
                #         rho = rho * 2
                #         flag = True
                #     elif (s_norm / eps_dual) > (1 / self.reltol / 1e2) * (r_norm / eps_primal):
                #         rho = rho / 2
                #         flag = True
                #     else:
                #         flag = False

                #     if flag:
                #         M = (M_constant + rho * (Acnorm ** 2)).unsqueeze(1)
                #         Minv = lambda x: x / M
                #         ATAfun = LPATA_Func(A, AT, rho, sigma)
                #         ATAop = LinearOp(A_fun=ATAfun, AT_fun=ATAfun, shape=(n, n))

                history['r_norm'].append(r_norm.item())
                history['s_norm'].append(s_norm.item())
                history['eps_primal'].append(eps_primal.item())
                history['eps_dual'].append(eps_dual.item())
                history['objval'].append(objval.item())

                if not self.training and r_norm < eps_primal and s_norm < eps_dual:
                    if self.verbose:
                        print(f'Obj: {objval.item():.2e}, res_primal: {r_norm.item():.2e}, res_dual: {s_norm.item():.2e}, eps_primal: {eps_primal.item():.2e}, eps_dual: {eps_dual.item():.2e}, rho: {float(rho):.2e}')
                    break

                if not self.training and self.verbose:
                    if k % 1000 == 0:
                        print(f'Obj: {objval.item():.2e}, res_primal: {r_norm.item():.2e}, res_dual: {s_norm.item():.2e}, eps_primal: {eps_primal.item():.2e}, eps_dual: {eps_dual.item():.2e}, rho: {float(rho):.2e}')

        if polish:
            x, z, y = self.solution_polishing(c, A, x, z, y)
        dz = z - torch.clip(z, min=lb, max=ub)
        res_z = vector_norm(dz)
        objval, r_norm, s_norm, eps_primal, eps_dual = self.eval_result(c, A, AT, d, e, gamma_c, gamma_b, x, z, y)

        print(f'Obj: {objval.item():.2e}, res_z: {res_z.item():.2e}, res_primal: {r_norm.item():.2e}, res_dual: {s_norm.item():.2e}, eps_primal: {eps_primal.item():.2e}, eps_dual: {eps_dual.item():.2e}, rho: {float(rho):.2e}')

        results = objval, r_norm, s_norm, eps_primal, eps_dual
        return x * d / gamma_b, history, results

    def solution_polishing(self, c, A, x, z, y):
        n = x.shape[0]
        device = c.device
        Il = (y < 0)[:, 0]
        Iu = (y > 0)[:, 0]
        zl = z[Il, 0]
        zu = z[Iu, 0]
        Al = A[Il, :]
        Au = A[Iu, :]
        delta = 1e-6

        # solution polishing matrix Kp
        Kp = torch.vstack([
            torch.hstack([delta * torch.eye(n, device=device), Al.T, Au.T]),
            torch.hstack([Al, -delta * torch.eye(Al.shape[0], device=device), torch.zeros((Al.shape[0], Au.shape[0]), device=device)]),
            torch.hstack([Au, torch.zeros((Au.shape[0], Al.shape[0]), device=device), -delta * torch.eye(Au.shape[0], device=device)])
        ])

        Kp_gt = torch.vstack([
            torch.hstack([torch.zeros((n, n), device=device), Al.T, Au.T]),
            torch.hstack([Al, torch.zeros((Al.shape[0], Al.shape[0]), device=device), torch.zeros((Al.shape[0], Au.shape[0]), device=device)]),
            torch.hstack([Au, torch.zeros((Au.shape[0], Al.shape[0]), device=device), torch.zeros((Au.shape[0], Au.shape[0]), device=device)])
        ])

        rhs = torch.concatenate([-c[:, 0], zl, zu]).unsqueeze_(1)
        Kpinv = torch.inverse(Kp)
        t_hat = Kpinv @ rhs

        for _ in range(2):
            dt = Kpinv @ (rhs - Kp_gt @ t_hat)
            t_hat = t_hat + dt

        x, yl, yu = torch.split(t_hat, [x.shape[0], zl.shape[0], zu.shape[0]])
        z = A @ x
        y[Il, :] = yl
        y[Iu, :] = yu
        return x, z, y

    def _solve_one_iter_precond(self, variables, c, A, AT, ATAop, Minv, lb, ub, rtol, rho, sigma, alpha, Kinv=None):
        x, z, y, xtilde = variables

        # x-update
        right = sigma * x - c + AT @ (rho * z - y)
        if Kinv is not None:
            xtilde = Kinv @ right
        else:
            # verbose = True if rtol < 1e-10 else False
            xtilde = pcg(ATAop, right, Minv=Minv, x0=xtilde.detach(), rtol=rtol, max_iters=int(1e3), verbose=False)
        ztilde = A @ (xtilde)
        x = alpha * xtilde + (1 - alpha) * x

        # z-update with relaxation
        ztilde = alpha * ztilde + (1 - alpha) * z
        z = torch.clip(ztilde + 1 / rho * y, min=lb, max=ub)

        # dual update
        y = y + rho * (ztilde - z)

        return x, z, y, xtilde

    def eval_result(self, c, A, AT, d, e, gamma_c, gamma_b, x, z, y):
        vector_norm = partial(torch.linalg.vector_norm, ord=self.norm_ord)
        m, n = A.shape
        objval = (c / d / gamma_c).T @ (x * d / gamma_b)
        Ax = A @ x
        ATy = AT @ y
        r_norm = vector_norm((Ax - z) / e / gamma_b)
        s_norm = vector_norm((c + ATy) / d / gamma_c)
        # eps_primal = self.abstol * (m**0.5) + self.reltol * max(vector_norm(Ax / e / gamma_b), vector_norm(z / e / gamma_b))
        # eps_dual = self.abstol * (n**0.5) + self.reltol * max(vector_norm(ATy / d / gamma_c),  vector_norm(c / d / gamma_c))
        eps_primal = self.abstol + self.reltol * max(vector_norm(Ax / e / gamma_b), vector_norm(z / e / gamma_b))
        eps_dual = self.abstol + self.reltol * max(vector_norm(ATy / d / gamma_c), vector_norm(c / d / gamma_c))
        return objval, r_norm, s_norm, eps_primal, eps_dual

    def extra_repr(self) -> str:
        # gamma_b_mul = torch.exp(self.gamma_b_mul_log).item()
        # gamma_c_mul = torch.exp(self.gamma_c_mul_log).item()
        # rho = torch.exp(self.rho_log).item()

        gamma_b_mul = self.gamma_b_mul.item()
        gamma_c_mul = self.gamma_c_mul.item()
        rho = self.rho.item()

        sigma = torch.exp(self.sigma_log).item()
        info = f"rho={rho:.3e}; sigma={sigma:.3e}; alpha={self.alpha.item():.3e}; gamma_c_mul={gamma_c_mul:.3e}; gamma_b_mul={gamma_b_mul:.3e};"
        if self.problem_scale is not None:
            d_mul_m = torch.exp(self.d_mul_log).mean()
            e_mul_m = torch.exp(self.e_mul_log).mean()
            info = info + f" e_mul_m={e_mul_m}; d_mul_m={d_mul_m}"
        return info
