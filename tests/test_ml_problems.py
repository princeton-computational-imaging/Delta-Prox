import numpy as np
import dprox as dp


def test_lsq():
    x = dp.Variable((3,3))
    rhs = np.array([[1, 2, 3],[4,5,6],[7,8,9]])
    prob = dp.Problem(dp.sum_squares(2*x - rhs))
    prob.solve('admm', x0=np.zeros((3,3)))
    print(x.value)

    assert (x.value.cpu().numpy() == rhs / 2).all()
    

def test_lsq1():
    x = dp.Variable((3,3))
    rhs = np.array([[1, 2, 3],[4,5,6],[7,8,9]])
    prob = dp.Problem(dp.sum_squares(2*x, rhs))
    prob.solve('admm', x0=np.zeros((3,3)))
    print(x.value)

    assert (x.value.cpu().numpy() == rhs / 2).all()
    
    
def test_lsq2():
    x = dp.Variable((3,3,1))
    rhs = np.array([[[1, 2, 3],[4,5,6],[7,8,9]]])
    kernel = np.array([[1,1],[1,1]]) / 4
    prob = dp.Problem(dp.sum_squares(dp.conv(x, kernel) - rhs))
    prob.solve('admm', x0=np.zeros((3,3,1)))
    out = dp.eval(dp.conv(x, kernel)-rhs, x.value, zero_out_constant=False)
    print(x.value)
    print(out)
    assert (out < 1e-5).all()
    
    
def test_lsq3():
    x = dp.Variable((3))
    rhs = np.array([1, 2, 3])
    prob = dp.Problem(dp.sum_squares(2*x - rhs))
    prob.solve('admm', x0=np.zeros(3))
    print(x.value)

    assert (x.value.cpu().numpy() == rhs / 2).all()
    