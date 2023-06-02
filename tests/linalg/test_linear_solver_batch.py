import torch
import dprox as dp
import numpy as np


P = np.random.rand(5, 5)
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


def test_cg():
    A2 = torch.from_numpy(A)
    def K(x): return A2 @ x
    x2 = torch.from_numpy(x)
    b2 = torch.from_numpy(offset)

    xhat1 = dp.linalg.solve.cg(K, b2)
    xhat3 = dp.linalg.solve.pcg(K, b2)

    print('conjugate_gradient')
    print(torch.mean(torch.abs(xhat1 - x2)).item())
    print(xhat1.numpy())
    assert torch.allclose(xhat1, x2, rtol=rtol)

    # print('conjugate_gradient2')
    # print(torch.mean(torch.abs(xhat2-x2)).item())
    # print(xhat2.numpy())
    # assert torch.allclose(xhat2, x2, rtol=rtol)

    print('PCG')
    print(torch.mean(torch.abs(xhat3 - x2)).item())
    print(xhat3.numpy())
    assert torch.allclose(xhat3, x2, rtol=rtol)
