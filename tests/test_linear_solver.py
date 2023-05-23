import torch
import dprox as dp
import numpy as np


P = np.random.rand(5,5)
I = np.eye(5)
mu = 0.01
A = P.T @ P + mu * I
x = np.random.rand(5)
offset = A @ x
rtol = 1e-8


class MatrixLinOp(dp.LinOp):
    def __init__(self, A):
        super().__init__()
        self.A = torch.nn.parameter.Parameter(A)

    def forward(self, x):
        return self.A @ x

    def adjoint(self, x):
        return self.A.T @ x


def test_gmres_scipy():
    from scipy.sparse.linalg import gmres, LinearOperator
    A_op = LinearOperator(shape=(5,5), matvec=lambda b: A@b)
    
    xhat, _ = gmres(A_op, offset)
    
    print('gmres scipy')
    print(x)
    print(xhat)
    
    print(np.mean(np.abs(xhat-x)))
    assert np.allclose(xhat, x, rtol=rtol)


def test_cg_scipy():
    from scipy.sparse.linalg import cg, LinearOperator
    A_op = LinearOperator(shape=(5,5), matvec=lambda b: A@b)
    
    xhat, _ = cg(A_op, offset)
    
    print('cg scipy')
    print(x)
    print(xhat)
    
    print(np.mean(np.abs(xhat-x)))
    assert np.allclose(xhat, x, rtol=rtol)
    
    
def test_cg():
    A2 = torch.from_numpy(A)
    K = lambda x: A2@x
    x2 = torch.from_numpy(x)
    b2 = torch.from_numpy(offset)
    
    xhat1 = dp.linalg.solve.conjugate_gradient(K, b2)
    # xhat2 = dp.proxfn.linalg.solve.conjugate_gradient2(K, b2)
    xhat3 = dp.linalg.solve.PCG(K, b2)
    
    
    print('conjugate_gradient')
    print(torch.mean(torch.abs(xhat1-x2)).item())
    print(xhat1.numpy())
    assert torch.allclose(xhat1, x2, rtol=rtol)
    
    # print('conjugate_gradient2')
    # print(torch.mean(torch.abs(xhat2-x2)).item())
    # print(xhat2.numpy())
    # assert torch.allclose(xhat2, x2, rtol=rtol)

    print('PCG')
    print(torch.mean(torch.abs(xhat3-x2)).item())
    print(xhat3.numpy())
    assert torch.allclose(xhat3, x2, rtol=rtol)


def test_plss():
    A2 = torch.from_numpy(A)
    # K = lambda x: A2@x
    K = MatrixLinOp(A2)
    x2 = torch.from_numpy(x)
    b2 = torch.from_numpy(offset)
    
    xhat1 = dp.linalg.solve.PLSS(K, b2)
    
    print('PLSS')
    print(torch.mean(torch.abs(xhat1-x2)).item())
    print(xhat1.detach().numpy())
    assert torch.allclose(xhat1, x2, rtol=rtol)


def test_minres():
    A2 = torch.from_numpy(A)
    # K = lambda x: A2@x
    K = MatrixLinOp(A2)
    x2 = torch.from_numpy(x)
    b2 = torch.from_numpy(offset)
    
    with torch.no_grad():
        xhat1 = dp.linalg.solve.MINRES(K, b2)
    
    print('MINRES')
    print(torch.mean(torch.abs(xhat1-x2)).item())
    print(xhat1.detach().numpy())
    assert torch.allclose(xhat1, x2, rtol=rtol)
