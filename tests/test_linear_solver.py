import torch
import dprox as dp
import numpy as np

P = np.random.rand(5,5)
A = P.T@P
x = np.random.rand(5)
offset = A@x
    
    
def test_gmres_scipy():
    from scipy.sparse.linalg import gmres, LinearOperator
    A_op = LinearOperator(shape=(5,5), matvec=lambda b: A@b)
    
    xhat, _ = gmres(A_op, offset)
    
    print('gmres scipy')
    print(x)
    print(xhat)
    
    print(np.mean(np.abs(xhat-x)))
    assert np.allclose(xhat, x, rtol=1e-3)


def test_cg_scipy():
    from scipy.sparse.linalg import cg, LinearOperator
    A_op = LinearOperator(shape=(5,5), matvec=lambda b: A@b)
    
    xhat, _ = cg(A_op, offset)
    
    print('cg scipy')
    print(x)
    print(xhat)
    
    print(np.mean(np.abs(xhat-x)))
    assert np.allclose(xhat, x, rtol=1e-3)
    
    
def test_cg():
    A2 = torch.from_numpy(A)
    K = lambda x: A2@x
    x2 = torch.from_numpy(x)
    b2 = torch.from_numpy(offset)
    
    xhat1 = dp.proxfn.linalg.solve.conjugate_gradient(K, b2)
    xhat2 = dp.proxfn.linalg.solve.conjugate_gradient2(K, b2)
    
    
    print('conjugate_gradient')
    print(torch.mean(torch.abs(xhat1-x2)).item())
    print(xhat1.numpy())
    assert torch.allclose(xhat1, x2, rtol=1e-3)
    
    print('conjugate_gradient2')
    print(torch.mean(torch.abs(xhat2-x2)).item())
    print(xhat2.numpy())
    assert torch.allclose(xhat2, x2, rtol=1e-3)



# def test_pcg():
#     A2 = torch.from_numpy(A)
#     K = lambda x: A2@x
#     x2 = torch.from_numpy(x)
#     b2 = torch.from_numpy(offset)
    
#     xhat1 = dp.proxfn.linalg.solve.PCG(K, b2)
    
#     print('precondition conjugate_gradient')
#     print(torch.mean(torch.abs(xhat1-x2)).item())
#     print(xhat1.numpy())
#     assert torch.allclose(xhat1, x2, rtol=1e-3)


# def test_plss():
#     A2 = torch.from_numpy(A)
#     K = lambda x: A2@x
#     x2 = torch.from_numpy(x)
#     b2 = torch.from_numpy(offset)
    
#     xhat1 = dp.proxfn.linalg.solve.PLSS(K, b2)
    
#     print('precondition conjugate_gradient')
#     print(torch.mean(torch.abs(xhat1-x2)).item())
#     print(xhat1.numpy())
#     assert torch.allclose(xhat1, x2, rtol=1e-3)