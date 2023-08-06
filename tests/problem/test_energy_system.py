import scipy.io

import dprox as dp
from dprox.utils.huggingface import load_path
from dprox.contrib.energy_system import load_simple_cep_model


def test():
    c, A_ub, A_eq, b_ub, b_eq = load_simple_cep_model()

    x = dp.Variable()
    prob = dp.Problem(c @ x, [A_ub @ x <= b_ub, A_eq @ x == b_eq])
    out = prob.solve(method='admm', adapt_params=True)
    print(out)

    # reference solution
    solution = scipy.io.loadmat(load_path("energy_system/simple_cep_model_20220916/esm_instance_solution.mat"))

    # pcg with custom backward
    # 83434.40576028604 170.29735095665677 0.0041508882261022985 200.8266268913533 0.569342163225932 29.714674732313096
    # tensor(-170.2974, device='cuda:0', dtype=torch.float64)

    # raw pcg
    # 83433.69636112433 174.7402594703876 0.004143616818155279 200.82662689427482 0.5693421632258844 29.691228047449876
    # tensor(-174.7403, device='cuda:0', dtype=torch.float64)

    # Obj: 8.35e+04, res_z: 0.00e+00, res_primal: 1.95e+02, reƒs_dual: 9.19e-04, eps_primal: 2.00e+02, eps_dual: 1.00e-03, rho: 1.45e+01
    # tensor(-99.6752, device='cuda:0', dtype=torch.float64)
