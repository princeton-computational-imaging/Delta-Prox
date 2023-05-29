import dprox as dp
from dprox.utils.examples.energy_system import load_simple_cep_model


def test():
    c, A_ub, A_eq, b_ub, b_eq = load_simple_cep_model()

    x = dp.Variable()
    prob = dp.Problem(c @ x, [A_ub @ x <= b_ub, A_eq @ x == b_eq])
    out = prob.solve(method='admm', adapt_params=True)
    print(out)
