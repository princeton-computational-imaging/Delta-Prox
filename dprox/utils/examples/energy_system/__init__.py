import numpy as np
from scipy import io

from dprox.utils.io import get_path


def load_simple_cep_model():
    model_components = io.loadmat(get_path("data/energy_system/simple_cep_model_20220916/esm_instance.mat"))
    n_con, n_var = model_components["A"].shape
    print("Number of linear constraints (w/o bound constraints):", n_con)
    print("Number of decision variables:", n_var)

    A = model_components["A"].astype(np.float64)
    b = model_components["rhs"].astype(np.float64)
    types = model_components["sense"]

    A_ub = A[types == '<']
    b_ub = b[types == '<'][:, 0]
    n1 = sum(types == '<')
    print('n1, A_ub, b_ub:', n1, A_ub.shape, b_ub.shape)

    A_eq = A[types == '=']
    b_eq = b[types == '='][:, 0]
    n2 = sum(types == '=')
    print('n2, A_eq, b_eq:', n2, A_eq.shape, b_eq.shape)
    assert n1 + n2 == n_con

    c = model_components["obj"][:, 0]
    print('c:', c.shape)

    return c, A_ub, A_eq, b_ub, b_eq
