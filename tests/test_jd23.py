import torch

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
    prob = Problem(data_term + reg_term, linear_solve_config=LinearSolveConfig(max_iters=50))

    max_iter = 5
    rhos, sigmas = log_descent(35, 30, max_iter)
    out = prob.solve(method='admm', x0=b, rhos=rhos, lams={reg_term: sigmas}, max_iter=max_iter, pbar=True)

    print(psnr(out, img))  # 25.25

    assert abs(psnr(out, img) - 25.25) < 0.1


def load(name='face', return_tensor=True):
    import imageio
    s = imageio.imread(name).copy().astype('float32') / 255
    s = s[:768,:,:]
    if return_tensor:
        s = to_torch_tensor(s, batch=True).float()
    return s


def test_jd2_batched():
    psf = point_spread_function(15, 5)
    
    img = load('tests/face2.png')
    # img = sample('face')
    img1 = to_torch_tensor(img, batch=True).float()
    b = blurring(img1, psf)
    b1 = mosaicing(b)
    
    img = sample('face')
    img2 = to_torch_tensor(img, batch=True).float()
    b = blurring(img2, psf)
    b2 = mosaicing(b)
    b = torch.cat([b1, b2], dim=0)
    print(b.shape)

    x = Variable()
    data_term = sum_squares(mosaic(conv(x, psf)) - b)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    prob = Problem(data_term + reg_term, linear_solve_config=LinearSolveConfig(max_iters=50))

    max_iter = 5
    rhos, sigmas = log_descent(35, 30, max_iter)
    out = prob.solve(method='admm', x0=b, rhos=rhos, lams={reg_term: sigmas}, max_iter=max_iter, pbar=True)

    print(psnr(out[0:1], img1))  # 29.689
    print(psnr(out[1].unsqueeze(0), img2))  # 29.689

    assert abs(psnr(out[0:1], img1) - 29.92) < 0.1
    assert abs(psnr(out[1].unsqueeze(0), img2) - 25.25) < 0.1