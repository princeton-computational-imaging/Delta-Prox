import torch
import torch.nn as nn
import dprox
import copy


seed = 123


class LinOp(dprox.LinOp):
    def __init__(self, A):
        super().__init__()
        self.A = nn.parameter.Parameter(A)

    def forward(self, x):
        return self.A @ x

    def adjoint(self, x):
        return self.A.T @ x


class SymmetricLinOp(dprox.LinOp):
    def __init__(self, P):
        super().__init__()
        self.P = nn.parameter.Parameter(P)

    def forward(self, x):
        return self.P.T @ self.P @ x

    def adjoint(self, x):
        return self.P.T @ self.P @ x


def test_linear_solver_torch_forward():
    torch.manual_seed(seed)

    P = torch.randn(5, 5)
    A = P.T @ P
    x = torch.randn(5)
    b = A @ x
    b = b.clone().detach().requires_grad_(True)

    A = LinOp(A)
    xhat = dprox.proxfn.linalg.linear_solve(A, b)

    print(torch.mean(torch.abs(xhat - x)))
    print(x)
    print(xhat)
    assert torch.allclose(x, xhat, rtol=1e-3)


def test_linear_solver_torch_backward_db():
    # gradient with implicit differentiation
    torch.manual_seed(seed)

    P = torch.randn(5, 5)
    A = P.T @ P
    x = torch.randn(5)
    b = A @ x
    b = b.clone().detach().requires_grad_(True)

    A = LinOp(A)

    xhat = dprox.proxfn.linalg.linear_solve(A, b)

    xhat.mean().backward()

    grad1 = b.grad

    # gradient with unrolling
    torch.manual_seed(seed)

    P = torch.randn(5, 5)
    A = P.T @ P
    x = torch.randn(5)
    b = A @ x
    b = b.clone().detach().requires_grad_(True)

    A = LinOp(A)

    xhat = dprox.proxfn.linalg.solve.conjugate_gradient(A, b)

    xhat.mean().backward()

    grad2 = b.grad

    print(grad1)
    print(grad2)

    assert torch.allclose(grad1, grad2, rtol=1e-3)


def test_linear_solver_torch_backward_dtheta():
    # gradient with implicit differentiation
    torch.manual_seed(seed)

    P = torch.randn(5, 5)
    A = SymmetricLinOp(P)

    x = torch.randn(5)
    with torch.no_grad():
        b = A(x)
    b = b.clone().detach().requires_grad_(True)

    xhat = dprox.proxfn.linalg.linear_solve(A, b)
    xhat.mean().backward()
    grad1 = A.P.grad

    # gradient with unrolling
    torch.manual_seed(seed)

    P = torch.randn(5, 5)
    A = SymmetricLinOp(P)

    x = torch.randn(5)
    with torch.no_grad():
        b = A(x)
    b = b.clone().detach().requires_grad_(True)

    xhat = dprox.proxfn.linalg.solve.conjugate_gradient(A, b)
    xhat.mean().backward()
    grad2 = A.P.grad

    # summary
    print('dtheta')
    print(grad1)
    print(grad2)

    print((grad1 - grad2).abs().max())
    assert torch.allclose(grad1, grad2, rtol=1e-2, atol=1e-2)


def test_linear_solver_torch_backward_dtheta2():
    # gradient with implicit differentiation
    torch.manual_seed(seed)

    P = torch.randn(5, 5)
    A = LinOp(P @ P.T)

    x = torch.randn(5)
    with torch.no_grad():
        b = A(x)
    b = b.clone().detach().requires_grad_(True)

    xhat = dprox.proxfn.linalg.linear_solve(A, b)
    xhat.mean().backward()
    grad1 = A.A.grad

    # gradient with unrolling
    # warning:
    torch.manual_seed(seed)

    P = torch.randn(5, 5)
    A = LinOp(P @ P.T)

    x = torch.randn(5)
    with torch.no_grad():
        b = A(x)
    b = b.clone().detach().requires_grad_(True)

    xhat = dprox.proxfn.linalg.solve.conjugate_gradient(A, b)
    xhat.mean().backward()
    grad2 = A.A.grad  # grad2 is only correct at the diagonal items.

    # explicit solve
    torch.manual_seed(seed)

    P = torch.randn(5, 5).requires_grad_(True)
    A = P @ P.T
    A.retain_grad()
    x = torch.randn(5)
    with torch.no_grad():
        b = A @ x
    b = b.clone().detach().requires_grad_(True)

    xhat = torch.linalg.solve(A, b)
    xhat.mean().backward()

    grad3 = A.grad

    # summary
    print('dtheta')
    print(grad1)
    print(grad2)
    print(grad3)

    print((grad1 - grad3).abs().max())
    assert torch.allclose(grad1, grad3, rtol=1e-2, atol=1e-2)
