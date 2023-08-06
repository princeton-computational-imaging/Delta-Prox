from dprox import *
from dprox.utils import *
from dprox import contrib


def test_admm():
    img = contrib.sample('face')
    psf = contrib.point_spread_function(15, 5)
    b = contrib.blurring(img, psf)

    x = Variable()
    data_term = sum_squares(conv(x, psf) - b)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    reg2 = nonneg(x)
    prob = Problem(data_term + reg_term + reg2)

    out = prob.solve(method='admm', x0=b, pbar=True)

    print(psnr(out, img))
    assert abs(psnr(out, img) - 34.51) < 0.1


def test_ladmm():
    img = contrib.sample('face')
    psf = contrib.point_spread_function(15, 5)
    b = contrib.blurring(img, psf)

    x = Variable()
    data_term = sum_squares(conv(x, psf) - b)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    reg2 = nonneg(x)
    prob = Problem(data_term + reg_term + reg2)

    out = prob.solve(method='ladmm', x0=b, pbar=True)

    print(psnr(out, img))
    assert abs(psnr(out, img) - 34.51) < 0.1


def test_admm_vxu():
    img = contrib.sample('face')
    psf = contrib.point_spread_function(15, 5)
    b = contrib.blurring(img, psf)

    x = Variable()
    data_term = sum_squares(conv(x, psf) - b)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    reg2 = nonneg(x)
    prob = Problem(data_term + reg_term + reg2)

    out = prob.solve(method='admm_vxu', x0=b, pbar=True)

    print(psnr(out, img))
    assert abs(psnr(out, img) - 34.50) < 0.1


def test_hqs():
    img = contrib.sample('face')
    psf = contrib.point_spread_function(15, 5)
    b = contrib.blurring(img, psf)

    x = Variable()
    data_term = sum_squares(conv(x, psf) - b)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    reg2 = nonneg(x)
    prob = Problem(data_term + reg_term + reg2)

    out = prob.solve(method='hqs', x0=b, pbar=True)

    print(psnr(out, img))
    assert abs(psnr(out, img) - 34.08) < 0.1


def test_pc():
    img = contrib.sample('face')
    psf = contrib.point_spread_function(15, 5)
    b = contrib.blurring(img, psf)

    x = Variable()
    data_term = sum_squares(conv(x, psf) - b)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    reg2 = nonneg(x)
    prob = Problem(data_term + reg_term + reg2)

    out = prob.solve(method='pc', x0=b, pbar=True)

    print(psnr(out, img))
    assert abs(psnr(out, img) - 29.87) < 0.1


def test_pgd():
    img = contrib.sample('face')
    psf = contrib.point_spread_function(15, 5)
    b = contrib.blurring(img, psf)

    x = Variable()
    data_term = sum_squares(conv(x, psf) - b)
    reg_term = deep_prior(x, denoiser='ffdnet_color')
    prob = Problem(data_term + reg_term)

    out = prob.solve(method='pgd', x0=b, pbar=True)

    print(psnr(out, img))
    assert abs(psnr(out, img) - 21.44) < 0.1
