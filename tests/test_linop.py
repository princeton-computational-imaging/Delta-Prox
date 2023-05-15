import torch
import dprox as dp

from dprox.utils.examples import fspecial_gaussian
from dprox.utils import to_torch_tensor

from scipy.misc import face


def test_conv():
    x = dp.Variable()
    psf = fspecial_gaussian(15, 5)
    op = dp.conv(x, psf)

    K = dp.CompGraph(op)
    assert K.sanity_check()

    img = to_torch_tensor(face(), batch=True)
    out = K.forward(img)

    print(img.shape)
    print(out.shape)
    assert img.shape == out.shape


def test_constant():
    x = dp.Variable()
    y = 3 * (x - torch.tensor([2, 2, 2]))

    input = torch.tensor([1, 2, 3])
    print(dp.eval(y, input, zero_out_constant=False))

    x.value = torch.tensor([1, 2, 3])

    print(y.variables)
    print(y.constants)
    print(y.value)
    print(y.offset)
    assert torch.allclose(y.value, 3 * (torch.tensor([1, 2, 3]) - torch.tensor([2, 2, 2])))
    assert torch.allclose(y.offset, - 3 * torch.tensor([2, 2, 2]))


def test_grad():
    x = dp.Variable()
    K = dp.CompGraph(dp.grad(x, dim=1) + dp.grad(x, dim=2))
    assert K.sanity_check()
    img = to_torch_tensor(face(), batch=True)
    print(img.shape)
    outputs = K.forward(img)


def test_mosaic():
    x = dp.Variable()
    K = dp.CompGraph(dp.mosaic(x))
    assert K.sanity_check()
    img = to_torch_tensor(face(), batch=True)
    print(img.shape)
    outputs = K.forward(img)


def test_sum():
    x1 = dp.Variable()
    x2 = dp.Variable()

    K = dp.CompGraph(x1 + x2)

    v1 = torch.randn((4, 4), requires_grad=True)
    v2 = torch.randn((4, 4))

    outputs = K.forward(v1, v2)

    print(outputs)
    assert torch.allclose(outputs, v1 + v2)

    loss = torch.mean(outputs)
    loss.backward()
    print(v1.grad)
    assert torch.allclose(v1.grad, torch.full_like(v1.grad, 1 / 16))


def test_variable():
    x = dp.Variable()

    print(x.forward(torch.tensor([2, 2, 2])))
    print(x.adjoint(torch.tensor([2, 2, 2])))

    K = dp.CompGraph(x)

    out = K.adjoint(torch.tensor([2, 2, 2]))
    print(out)


def test_vstack():
    x = dp.Variable()
    K = dp.CompGraph(dp.vstack([dp.mosaic(x), dp.grad(x)]))

    img = to_torch_tensor(face(), batch=True)
    print(img.shape)

    outputs = K.forward(img)
    inputs = K.adjoint(outputs)
    print(inputs.shape)
    K.sanity_check()


def test_complex():
    from dprox.utils import examples

    img = examples.sample('face')
    psf = examples.point_spread_function(15, 5)
    b = examples.blurring(img, psf)

    x = dp.Variable()
    data_term = dp.sum_squares(dp.conv(x, psf) - b)
    reg_term = dp.deep_prior(x, denoiser='ffdnet_color')
    reg2 = dp.nonneg(x)
    K = dp.CompGraph(dp.vstack([fn.linop for fn in [data_term, reg_term, reg2]]))
    K.forward(b)
