from dprox import *
from dprox.contrib.optic import *
from dprox.utils import *


def test_deep_prior():
    device = torch.device("cuda")
    x = Variable()
    reg_term = deep_prior(x, denoiser="ffdnet_color").to(device)

    sigma = torch.nn.Parameter(torch.tensor(0.1).to(device))
    inp = torch.randn((1, 3, 128, 128), device=device)
    x.value = inp
    y = reg_term.prox(inp, sigma)

    loss = torch.nn.functional.mse_loss(inp, y)
    loss.backward()
    print(sigma.grad)
