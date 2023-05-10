import torch
import dprox


seed = 123


class LinOp(torch.nn.Module):
    def __init__(self, A) -> None:
        super().__init__()
        self.A = A

    def forward(self, x):
        return self.A @ x

    def adjoint(self, x):
        return self.A.T @ x

    @property
    def T(self):
        return LinOp(self.A.T)

    @property
    def params(self):
        return [self.A]


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
