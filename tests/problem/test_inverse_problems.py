import numpy as np

from dprox import *
from dprox.utils import *
from dprox.utils import examples


def test_csmri():
    x0, y0, gt, mask = examples.csmri.sample()

    x = Variable()
    y = Placeholder()
    data_term = csmri(x, mask, y)
    reg_term = deep_prior(x, denoiser='unet')
    prob = Problem(data_term + reg_term)

    y.value = y0
    max_iter = 24
    rhos, sigmas = log_descent(30, 20, max_iter)
    prob.solve(
        method='admm',
        device='cuda',
        x0=x0, rhos=rhos, lams={reg_term: sigmas}, max_iter=max_iter, pbar=True
    )
    out = x.value.real

    print(psnr(out, gt))
    assert abs(psnr(out, gt) - 42.6) < 0.1


def test_deconv():
    img = examples.sample('face')
    psf = examples.point_spread_function(15, 5)
    b = examples.blurring(img, psf)

    x = Variable()
    data_term = sum_squares(conv(x, psf) - b)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    reg2 = nonneg(x)
    prob = Problem(data_term + reg_term + reg2)

    out = prob.solve(method='admm', x0=b, pbar=True)

    print(psnr(out, img))
    assert abs(psnr(out, img) - 31.9) < 0.1


def test_deconv2():
    img = examples.sample('face')
    psf = examples.point_spread_function(ksize=15, sigma=5)
    # TODO: this still has bug
    y = examples.blurring(img, psf) + np.random.randn(*img.shape).astype('float32') * 5 / 255.0
    y.squeeze(0)
    print(img.shape, y.shape)

    x = Variable()
    data_term = sum_squares(conv(x, psf) - y)
    prior_term = deep_prior(x, 'ffdnet_color')
    reg_term = nonneg(x)
    objective = data_term + prior_term + reg_term
    p = Problem(objective)
    out = p.solve(method='admm', x0=y, pbar=True)
    print(psnr(out, img))
