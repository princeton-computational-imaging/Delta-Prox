import dprox as dp
from dprox.utils.examples.energy_system import load_simple_cep_model


def test():
    c, A_ub, A_eq, b_ub, b_eq = load_simple_cep_model()

    x = dp.Variable()
    prob = dp.Problem(c @ x, [A_ub @ x <= b_ub, A_eq @ x == b_eq])
    out = prob.solve(method='admm', adapt_params=True)
    print(out)
    # pcg with custom backward
    # 83434.40576028604 170.29735095665677 0.0041508882261022985 200.8266268913533 0.569342163225932 29.714674732313096
    # tensor(-170.2974, device='cuda:0', dtype=torch.float64)   
    
    # raw pcg
    # 83433.69636112433 174.7402594703876 0.004143616818155279 200.82662689427482 0.5693421632258844 29.691228047449876
    # tensor(-174.7403, device='cuda:0', dtype=torch.float64)
