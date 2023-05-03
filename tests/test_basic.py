import numpy as np
from dprox import *

def test_lsq():
    x = Variable((3,3))
    rhs = np.array([[1, 2, 3],[4,5,6],[7,8,9]])
    prob = Problem(sum_squares(2*x - rhs))
    prob.solve('admm', x0=np.zeros((3,3)))
    print(x.value)

    assert (x.value.cpu().numpy() == rhs / 2).all()