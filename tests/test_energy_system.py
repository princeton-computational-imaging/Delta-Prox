import dprox as dp


def test():
    x = dp.Variable()
    prob = dp.Problem(c @ x, [A_ub @ x <= b_ub, A_eq @ x == b_eq])
    out = prob.solve(method='admm', adapt_params=True)