import torch
from functools import partial


atol, rtol = 1e-6, 1e-3
allclose = partial(torch.allclose, atol=atol, rtol=rtol)


def tol(input, other):
    tmp = torch.abs(input-other) - atol - rtol * torch.abs(other)
    return torch.max(tmp).item()


def auto_diff(seed):
    torch.manual_seed(seed)
    theta = torch.randn((32,32), requires_grad=True)
    K = theta * 2
    x = torch.randn((32))
    b = K @ x
    b = b.clone().detach().requires_grad_(True)

    xhat = torch.linalg.solve(K, b)

    loss = xhat.mean()
    loss.backward()

    return theta.grad, b.grad


def differentiate_dloss_db(seed):
    torch.manual_seed(seed)
    theta = torch.randn((32,32), requires_grad=True)
    K = theta * 2
    x = torch.randn((32))
    b = K @ x
    b = b.clone().detach().requires_grad_(True)

    xhat = torch.linalg.solve(K, b)
    xhat.retain_grad()

    loss = xhat.mean()
    loss.backward()

    db_matrix = torch.inverse(K.T) @ xhat.grad
    db_matrix_free = torch.linalg.solve(K.T, xhat.grad)
    
    return db_matrix, db_matrix_free
    
    
def matrix_differentiate_dloss_dtheta(seed):
    torch.manual_seed(seed)
    theta = torch.randn((32,32), requires_grad=True)
    K = theta * 2
    x = torch.randn((32))
    b = K @ x
    b = b.clone().detach().requires_grad_(True)

    xhat = torch.linalg.solve(K, b)
    xhat.retain_grad()

    loss = xhat.mean()
    loss.backward()
    
    
    def Kmat(theta):
        return theta * 2

    dK_dtheta = torch.autograd.functional.jacobian(Kmat, theta)
    dxhat_dtheta = -torch.inverse(K) @ dK_dtheta @ xhat

    dloss_dtheta = dxhat_dtheta @ xhat.grad

    return dloss_dtheta

    
def matrix_free_differentiate_dloss_dtheta(seed):
    torch.manual_seed(seed)
    theta = torch.randn((32,32), requires_grad=True)
    K = theta * 2
    x = torch.randn((32))
    b = K @ x
    b = b.clone().detach().requires_grad_(True)

    xhat = torch.linalg.solve(K, b)
    xhat.retain_grad()

    loss = xhat.mean()
    loss.backward()
    
    def linop(theta):
        return theta*2 @ xhat

    db_dtheta = -torch.autograd.functional.jacobian(linop, theta).permute(1,2,0).unsqueeze(-1)
    dxhat_dtheta = torch.linalg.solve(K, db_dtheta).squeeze(-1)

    dloss_dtheta = dxhat_dtheta @ xhat.grad

    return dloss_dtheta


def test_db():
    for seed in range(20):
        _, db_ref = auto_diff(seed)
        db_matrix, db_matrix_free = differentiate_dloss_db(seed)
        print('b', seed, torch.mean(torch.abs(db_ref- db_matrix)))
        print('b free', seed, torch.mean(torch.abs(db_ref- db_matrix_free)))
        assert allclose(db_matrix, db_ref)
        assert allclose(db_matrix_free, db_ref)
        
        
def test_theta():
    for seed in range(20):
        dtheta_ref, _ = auto_diff(seed)
        dtheta_matrix = matrix_differentiate_dloss_dtheta(seed)
        dtheta_matrix_free = matrix_free_differentiate_dloss_dtheta(seed)
        
        print('theta', seed, tol(dtheta_matrix, dtheta_ref))
        print('theta free', seed, tol(dtheta_matrix_free, dtheta_ref), 
              dtheta_ref.abs().max().item(), 
              (dtheta_matrix_free-dtheta_ref).abs().max().item())
        
        assert allclose(dtheta_matrix, dtheta_ref)
        assert allclose(dtheta_matrix_free, dtheta_ref)
        
        