from dprox.proxfn import least_squares, ext_sum_squares, least_squares
from dprox.linop import Variable


def get_least_square_solver(psi_fns, omega_fns, try_diagonalize, try_freq_diagonalize, linear_solve_config):
    prox_fns = psi_fns + omega_fns

    ext_sq = [fn for fn in omega_fns if isinstance(fn, ext_sum_squares)]
    for fn in ext_sq:
        other = [f for f in prox_fns if f is not fn]
        if all(isinstance(fn.linop, Variable) for fn in other):
            return ext_sq[0].setup([f.b for f in omega_fns if f is not fn and f not in ext_sq])

    return least_squares(omega_fns, psi_fns, try_diagonalize, try_freq_diagonalize, linear_solve_config=linear_solve_config)
