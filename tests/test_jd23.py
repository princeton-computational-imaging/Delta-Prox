from dprox import *
from dprox.utils import *
from dprox.utils.examples import *
from dprox.linalg import LinearSolveConfig

def test_jd2():
    img = sample('face')
    img = to_torch_tensor(img, batch=True).float()
    psf = point_spread_function(15, 5)
    b = blurring(img, psf)
    b = mosaicing(b)

    x = Variable()
    data_term = sum_squares(mosaic(conv(x, psf)) - b)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    prob = Problem(data_term + reg_term, linear_solve_config=LinearSolveConfig(max_iters=500))

    max_iter = 24
    rhos, sigmas = log_descent(35, 30, max_iter)
    out = prob.solve(method='admm', x0=b, rhos=rhos, lams={reg_term: sigmas}, max_iter=24, pbar=True)

    imshow(out)
    print(psnr(out, img))  # 29.689

    assert psnr(out, img) - 29.9 < 0.1
