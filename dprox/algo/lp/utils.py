from functools import partial

import numpy as np
import torch
from scipy import io, sparse
from scipy.sparse import linalg as slinalg
from tqdm import tqdm


class LPATA_Func(torch.nn.Module):
    def __init__(self, A, AT, rho, sigma) -> None:
        super().__init__()
        self.A = A
        self.AT = AT
        self.rho = rho
        self.sigma = sigma

    def forward(self, x):
        tmp = (self.A @ x) * self.rho
        out = (self.AT @ tmp) + self.sigma * x
        return out


class LinearOp(torch.nn.Module):
    def __init__(self, A_fun, AT_fun, shape) -> None:
        super().__init__()
        self.A_fun = A_fun
        self.AT_fun = AT_fun
        self.shape = shape

    def forward(self, x):
        return self.A_fun(x)

    def adjoint(self, x):
        return self.AT_fun(x)

    def __matmul__(self, x):
        return self.forward(x)

    def transpose(self):
        """Transpose this linear operator.
        Returns a LinearOperator that represents the transpose of this one.
        Can be abbreviated self.T instead of self.transpose().
        """
        return LinearOp(A_fun=self.AT_fun, AT_fun=self.A_fun, shape=(self.shape[1], self.shape[0]))

    T = property(transpose)


def Ruiz_equilibration_th(A, c, b, ord=float('inf'), max_iters=20, max_tolerance=1e1, verbose=True):
    vector_norm = partial(torch.linalg.vector_norm, ord=ord)
    MAX_ITER = max_iters
    device = A.device
    m, n = A.shape
    eps_equil = 1e-3
    A_bar = A
    c_bar = c

    e = torch.ones((m, 1), device=device)
    d = torch.ones((1, n), device=device)

    coeff = (m / n) ** (1 / 2 / ord) if ord < float('inf') else 1

    # for _ in tqdm(range(MAX_ITER)):
    for _ in (range(MAX_ITER)):
        delta1 = 1 / torch.linalg.norm(A_bar, ord=ord, dim=0) ** 0.5
        delta2 = 1 / torch.linalg.norm(A_bar, ord=ord, dim=1) ** 0.5
        d = d * delta1.view(1, n) * coeff
        e = e * delta2.view(m, 1)
        # A_bar = E @ A @ D
        A_bar = A * e * d

        if max(vector_norm(1 - delta1, ord=float('inf')),
               vector_norm(1 - delta2, ord=float('inf'))) < eps_equil:
            break

        # if ord < float('inf') and (e.max() / d.max() > max_tolerance or d.max() / e.max() > max_tolerance):
        #     break

    c_bar = c * d.view(n, 1)
    Arnorm = torch.linalg.norm(A_bar, ord=ord, dim=1)
    Acnorm = torch.linalg.norm(A_bar, ord=ord, dim=0)

    b_bar = b * e.view(m, 1)[:b.shape[0]]
    gamma_c = 1 / vector_norm(c_bar) * Arnorm.mean()
    gamma_b = 1 / vector_norm(b_bar) * Acnorm.mean()

    if verbose:
        print(Acnorm.max(), Acnorm.mean())
        print(Arnorm.max(), Arnorm.mean())
        print(d.max())
        print(e.max())

    return d, e, gamma_c, gamma_b, A_bar


def Ruiz_equilibration_sparse_np(A, c, b, ord=float('inf'), max_iters=100, max_tolerance=1e1, verbose=True):
    vector_norm = partial(np.linalg.norm, ord=ord)
    MAX_ITER = max_iters
    m, n = A.shape
    eps_equil = 1e-3
    A_bar = A

    d = np.ones(n)
    e = np.ones(m)

    coeff = (m / n) ** (1 / 2 / ord) if ord < float('inf') else 1

    # for _ in tqdm(range(MAX_ITER)):
    for _ in (range(MAX_ITER)):
        delta1 = 1 / slinalg.norm(A_bar, ord=ord, axis=0) ** 0.5
        delta2 = 1 / slinalg.norm(A_bar, ord=ord, axis=1) ** 0.5
        d = d * delta1.flatten() * coeff
        e = e * delta2.flatten()

        D = sparse.diags(d, 0)
        E = sparse.diags(e, 0)
        A_bar = E @ A @ D

        if max(vector_norm(1 - delta1, ord=float('inf')),
               vector_norm(1 - delta2, ord=float('inf'))) < eps_equil:
            break

        # if ord < float('inf') and (e.max() / d.max() > max_tolerance or d.max() / e.max() > max_tolerance):
        #     break

    c_bar = c * d
    Arnorm = slinalg.norm(A_bar, ord=ord, axis=1)
    Acnorm = slinalg.norm(A_bar, ord=ord, axis=0)

    b_bar = b * e[:b.shape[0]]
    gamma_c = 1 / vector_norm(c_bar) * Arnorm.mean()
    gamma_b = 1 / vector_norm(b_bar) * Acnorm.mean()
    # gamma_b = 1

    if verbose:
        print(Acnorm.max(), Acnorm.mean())
        print(Arnorm.max(), Arnorm.mean())
        print(d.max())
        print(e.max())

    return d, e, gamma_c, gamma_b, A_bar


def scipy_sparse_to_torchop(A, device=None, output_AT=True):
    A = sparse.coo_matrix(A)
    A_th = torch.sparse_coo_tensor(np.vstack([A.row, A.col]), A.data, size=A.shape).to_sparse_csr().to(device)
    if output_AT:
        AT_th = torch.sparse_coo_tensor(np.vstack([A.T.row, A.T.col]), A.T.data, size=A.T.shape).to_sparse_csr().to(device)
        return A_th, AT_th
    else:
        return A_th


def load_simple_cep_model():
    model_components = io.loadmat("simple_cep_model_20220916/output/esm_instance.mat")
    n_con, n_var = model_components["A"].shape
    print("Number of linear constraints (w/o bound constraints):", n_con)
    print("Number of decision variables:", n_var)

    A = model_components["A"].astype(np.float64)
    b = model_components["rhs"].astype(np.float64)
    types = model_components["sense"]

    A_ub = A[types == '<']
    b_ub = b[types == '<'][:, 0]
    n1 = sum(types == '<')
    print('n1, A_ub, b_ub:', n1, A_ub.shape, b_ub.shape)

    A_eq = A[types == '=']
    b_eq = b[types == '='][:, 0]
    n2 = sum(types == '=')
    print('n2, A_eq, b_eq:', n2, A_eq.shape, b_eq.shape)
    assert n1 + n2 == n_con

    c = model_components["obj"][:, 0]
    print('c:', c.shape)

    # lb = model_components["lb"][:,0]
    # ub = model_components["ub"][:,0]
    return c, A_ub, A_eq, b_ub, b_eq


def adjust_lr_cosine(optimizer, iteration, num_iters, base_lr, min_lr):
    ratio = iteration / num_iters
    lr = min_lr + (base_lr - min_lr) * (1.0 + np.cos(np.pi * ratio)) / 2.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == '__main__':
    pass
