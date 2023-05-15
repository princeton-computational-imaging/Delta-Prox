from dprox import *
from dprox.utils import *
from dprox.utils import examples


def test_admm():
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


def test_ladmm():
    img = examples.sample('face')
    psf = examples.point_spread_function(15, 5)
    b = examples.blurring(img, psf)

    x = Variable()
    data_term = sum_squares(conv(x, psf) - b)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    reg2 = nonneg(x)
    prob = Problem(data_term + reg_term + reg2)

    out = prob.solve(method='ladmm', x0=b, pbar=True)

    print(psnr(out, img))
    assert abs(psnr(out, img) - 31.9) < 0.1


def test_admm_vxu():
    img = examples.sample('face')
    psf = examples.point_spread_function(15, 5)
    b = examples.blurring(img, psf)

    x = Variable()
    data_term = sum_squares(conv(x, psf) - b)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    reg2 = nonneg(x)
    prob = Problem(data_term + reg_term + reg2)

    out = prob.solve(method='admm_vxu', x0=b, pbar=True)

    print(psnr(out, img))
    assert abs(psnr(out, img) - 31.9) < 0.1


def test_hqs():
    img = examples.sample('face')
    psf = examples.point_spread_function(15, 5)
    b = examples.blurring(img, psf)

    x = Variable()
    data_term = sum_squares(conv(x, psf) - b)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    reg2 = nonneg(x)
    prob = Problem(data_term + reg_term + reg2)

    out = prob.solve(method='hqs', x0=b, pbar=True)

    print(psnr(out, img))
    assert abs(psnr(out, img) - 31.6) < 0.1


def test_pc():
    img = examples.sample('face')
    psf = examples.point_spread_function(15, 5)
    b = examples.blurring(img, psf)

    x = Variable()
    data_term = sum_squares(conv(x, psf) - b)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    reg2 = nonneg(x)
    prob = Problem(data_term + reg_term + reg2)

    out = prob.solve(method='pc', x0=b, pbar=True)

    print(psnr(out, img))
    assert abs(psnr(out, img) - 29.6) < 0.1


def test_pgd():
    img = examples.sample('face')
    psf = examples.point_spread_function(15, 5)
    b = examples.blurring(img, psf)

    x = Variable()
    data_term = sum_squares(conv(x, psf) - b)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    prob = Problem(data_term + reg_term)

    out = prob.solve(method='pgd', x0=b, pbar=True)

    print(psnr(out, img))
