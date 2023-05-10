import torch
import dprox as dp
import torch.nn as nn

class LinOp(torch.nn.Module):
    def __init__(self, A) -> None:
        super().__init__()
        self.A = nn.parameter.Parameter(A).double() 
    def forward(self, x):
        return self.A@x.double()
    
    def adjoint(self, x):
        return self.A.T@x
    
    def T(self):
        return LinOp(self.A.T)
    
    def params(self):
        return [self.A]
    
    
def test_cg():
    P = torch.randn(5,5)
    A = P.T@P
    K = lambda x: A@x
    x = torch.randn(5)
    b = A@x
    
    xhat1 = dp.proxfn.linear_solve.conjugate_gradient(K, b)
    xhat2 = dp.proxfn.linear_solve.conjugate_gradient2(K, b)
    
    print(x)
    
    print(torch.mean(torch.abs(xhat1-x)))
    print(xhat1)
    assert torch.allclose(xhat1, x, rtol=1e-3)
    
    print(torch.mean(torch.abs(xhat2-x)))
    print(xhat2)
    assert torch.allclose(xhat2, x, rtol=1e-3)
  