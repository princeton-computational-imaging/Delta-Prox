from .qrnn3d import QRNNREDC3D
from .grunet import GRUnet

"""Define commonly used architecture"""


def qrnn3d():
    net = QRNNREDC3D(1, 16, 5, [1, 3], has_ad=True, bn=True)
    net.use_2dconv = False
    net.bandwise = False
    return net


def qrnn2d():
    net = QRNNREDC3D(1, 16, 5, [1, 3], has_ad=True, is_2d=True)
    net.use_2dconv = False
    net.bandwise = False
    return net


def qrnn3d_masked():
    net = QRNNREDC3D(2, 16, 5, [1, 3], has_ad=True)
    net.use_2dconv = False
    net.bandwise = False
    return net


def grunet_masked():
    return GRUnet(in_ch=2, out_ch=1, use_noise_map=True)


def grunet_masked_nobn():
    return GRUnet(in_ch=2, out_ch=1, use_noise_map=True, bn=False)


def grunet():
    return GRUnet(in_ch=1, out_ch=1, use_noise_map=False)


def grunet_nobn():
    return GRUnet(in_ch=1, out_ch=1, use_noise_map=False, bn=False)
