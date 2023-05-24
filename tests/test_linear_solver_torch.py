import torch
import torch.nn as nn
import dprox


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
    
    dtype = torch.float64
    P = torch.randn(5, 5, dtype=dtype)
    I = torch.eye(5, dtype=dtype)
    mu = 0.5
    A = P.T @ P + mu * I
    x = torch.randn(5, dtype=dtype)
    b = A @ x
    b = b.clone().detach().requires_grad_(True)

    A = LinOp(A)
    # xhat = dprox.linalg.linear_solve(A, b)
    # xhat = dprox.linalg.solve.conjugate_gradient(A, b, max_iters=100)
    xhat = dprox.linalg.solve.PCG(A, b, max_iters=100)    

    print(xhat.dtype)
    print(torch.mean(torch.abs(xhat - x)))
    print(x)
    print(xhat)
    assert torch.allclose(x, xhat, rtol=1e-6)


def test_linear_solver_torch_backward_db():
    # gradient with implicit differentiation
    torch.manual_seed(seed)

    P = torch.randn(5, 5)
    A = P.T @ P
    x = torch.randn(5)
    b = A @ x
    b = b.clone().detach().requires_grad_(True)

    A = LinOp(A)

    xhat = dprox.linalg.linear_solve(A, b)

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

    xhat = dprox.linalg.solve.conjugate_gradient(A, b)

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

    xhat = dprox.linalg.linear_solve(A, b)
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

    xhat = dprox.linalg.solve.conjugate_gradient(A, b)
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

    dtype = torch.float64
    P = torch.randn(5, 5, dtype=dtype)
    I = torch.eye(5, dtype=dtype)
    mu = 0.5
    A = LinOp(P @ P.T + mu * I)

    x = torch.randn(5, dtype=dtype)
    with torch.no_grad():
        b = A(x)
    b = b.clone().detach().requires_grad_(True)

    xhat = dprox.linalg.linear_solve(A, b)
    xhat.mean().backward()
    grad1 = A.A.grad

    # gradient with unrolling
    # warning:
    torch.manual_seed(seed)

    P = torch.randn(5, 5, dtype=dtype)
    A = LinOp(P @ P.T + mu * I)

    x = torch.randn(5, dtype=dtype)
    with torch.no_grad():
        b = A(x)
    b = b.clone().detach().requires_grad_(True)

    # xhat = dprox.linalg.solve.conjugate_gradient(A, b)
    xhat = dprox.linalg.solve.PCG(A, b, rtol=1e-6, max_iters=100, verbose=False)
    xhat.mean().backward()
    grad2 = A.A.grad  # grad2 is only correct at the diagonal items.

    # explicit solve
    torch.manual_seed(seed)

    P = torch.randn(5, 5, dtype=dtype).requires_grad_(True)
    A = P @ P.T + mu * I
    A.retain_grad()
    x = torch.randn(5, dtype=dtype)
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


def test_linear_solver_torch_forward_dconv_doe():
    torch.manual_seed(seed)
    x = dprox.Variable((1,1,10,10))
    psf = torch.randn((1,5,5))
    KtK = dprox.conv_doe(x, psf=psf).gram
    
    x = torch.randn(1,1,10,10)
    b = KtK(x)
    
    xhat = dprox.linalg.linear_solve(KtK, b)
    
    print(x.squeeze()[0])
    print(xhat.squeeze()[0])
    print((xhat-x).abs().mean())
    
    assert torch.allclose(xhat, x, atol=0.5, rtol=1)
    
    
def test_linear_solver_torch_backward_dconv_doe():
    torch.manual_seed(seed)
    x = dprox.Variable((1,1,10,10))
    psf = torch.randn((1,5,5))
    KtK = dprox.conv_doe(x, psf=psf).gram
    
    x = torch.randn(1,1,10,10)
    b = KtK(x)
    
    xhat = dprox.linalg.linear_solve(KtK, b)
    
    xhat.mean().backward()
    
    grad1 = KtK.psf.grad
    print(KtK.psf.grad)
    print(x.squeeze()[0])
    print(xhat.squeeze()[0])
    print((xhat-x).abs().mean())
    
    torch.manual_seed(seed)
    x = dprox.Variable((1,1,10,10))
    psf = torch.randn((1,5,5))
    KtK = dprox.conv_doe(x, psf=psf).gram
    
    x = torch.randn(1,1,10,10)
    b = KtK(x)
    
    xhat = dprox.linalg.solve.conjugate_gradient(KtK, b)
    
    xhat.mean().backward()
    
    grad2 = KtK.psf.grad
    print(KtK.psf.grad)
    print(x.squeeze()[0])
    print(xhat.squeeze()[0])
    print((xhat-x).abs().mean())
    
    print('########')
    print((grad1-grad2).abs().mean())
    
    
def test_linear_solver_torch_backward_dconv_doe_ktk():
    torch.manual_seed(seed)
    x = dprox.Variable((1,1,10,10))
    psf = torch.randn((1,5,5))
    KtK = dprox.conv_doe(x, psf=psf).gram
    
    x = torch.randn(1,1,10,10)
    b = KtK(x)
    
    xhat = dprox.linalg.linear_solve(KtK, b)
    
    xhat.mean().backward()
    
    print(KtK.psf.grad)
    # TODO: need a finite difference method to check gradient
    