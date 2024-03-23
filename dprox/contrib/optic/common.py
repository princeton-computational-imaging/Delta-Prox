from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_coordinate(nx, ny, dx, dy):
    """
    return the coordinates of a 2D grid with given dimensions and spacing.

    :param nx: The number of points in the x-direction
    :param ny: The number of points along the y-axis in a 2D grid
    :param dx: The spacing between two adjacent points along the x-axis
    :param dy: The spacing between the y-coordinates of the grid points. 
    :return: two tensors `xx` and `yy` which represent the x and y
    coordinates of a 2D grid. The size of the grid is determined by the input parameters `nx` and `ny`,
    and the spacing between grid points is determined by `dx` and `dy`.
    """
    x = (torch.arange(nx) - (nx - 1.) / 2) * dx
    y = (torch.arange(ny) - (ny - 1.) / 2) * dy
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    return xx, yy


def area_downsampling(input, target_side_length):
    """
    downsample an input image to a target side length by averaging the pixel values in
    each block of pixels.

    :param input: a 3-dimensional tensor representing an image, with dimensions (channels, height,
    width)
    :param target_side_length: The target side length is the desired length of one side of the output
    image after downsampling
    :return: the downsampled version of the input image with the target side length using average
    pooling.
    """
    if not input.shape[2] % target_side_length:
        factor = int(input.shape[2] / target_side_length)
        output = F.avg_pool2d(input=input, kernel_size=[factor, factor], stride=[factor, factor])
    else:
        raise NotImplementedError
    return output


def psf2otf(psf, output_size):
    """
    take a point spread function and output size as inputs, pad the PSF with zeros to
    match the output size, circularly shifts the PSF so the center pixel is at 0,0, and returns the
    optical transfer function.

    :param psf: The point spread function (PSF) is a 4-dimensional tensor representing the blur kernel
    used in image processing. It is typically used in image deconvolution to recover the original image
    from a blurred image
    :param output_size: The desired size of the output Optical Transfer Function (OTF) after padding the
    Point Spread Function (PSF) with zeros. It is a tuple of the form (batch_size, num_channels, height,
    width)
    :return: the optical transfer function (OTF) of the point spread function (PSF) after padding and
    circularly shifting the PSF.
    """
    _, _, fh, fw = psf.shape

    # pad out to output_size with zeros
    if output_size[2] != fh:
        pad = (output_size[2] - fh) / 2

        if (output_size[2] - fh) % 2 != 0:
            pad_top = pad_left = int(np.ceil(pad))
            pad_bottom = pad_right = int(np.floor(pad))
        else:
            pad_top = pad_left = int(pad) + 1
            pad_bottom = pad_right = int(pad) - 1

        padded = F.pad(input=psf, pad=[pad_left, pad_right, pad_top, pad_bottom], mode="constant")
    else:
        padded = psf

    # circularly shift so center pixel is at 0,0
    padded = torch.fft.ifftshift(padded)
    otf = torch.fft.fft2(padded)
    return otf


def img_psf_conv(img, psf, circular=True):
    """
    perform image convolution with a point spread function (PSF) and can handle both
    circular and linearized convolutions.

    :param img: a 4-dimensional tensor representing an image, with dimensions (batch_size, channels,
    height, width)
    :param psf: The point spread function (PSF) is a 4-dimensional tensor representing the blur kernel
    used in image processing. It is typically used in image deconvolution to recover the original image
    from a blurred image
    :param circular: A boolean parameter that determines whether the convolution should be circular or
    linearized. If circular is True, then circular convolution is performed. If circular is False, then
    linearized convolution is performed, defaults to False (optional)
    :return: the result of the convolution of the input image with the point spread function (PSF). If
    the circular parameter is set to False, the function performs a linearized convolution by padding
    the image and then cropping the result. The output is a tensor representing the convolved image.
    """
    if not circular:  # linearized conv
        target_side_length = 2 * img.shape[2]
        height_pad = (target_side_length - img.shape[2]) / 2
        width_pad = (target_side_length - img.shape[3]) / 2
        pad_top, pad_bottom = int(np.ceil(height_pad)), int(np.floor(height_pad))
        pad_left, pad_right = int(np.ceil(width_pad)), int(np.floor(width_pad))
        img = F.pad(input=img, pad=[pad_left, pad_right, pad_top, pad_bottom], mode="constant")

    img_fft = torch.fft.fft2(img)
    otf = psf2otf(psf, output_size=img.shape)
    result = torch.fft.ifft2(img_fft * otf)
    result = result.real

    if not circular:
        result = result[:, :, pad_top:-pad_bottom, pad_left:-pad_right]

    return result


class FresnelPropagator(nn.Module):
    def __init__(self, input_shape, distance, discretization_size, wave_lengths):
        super().__init__()
        self.input_shape = input_shape
        self.distance = distance
        self.wave_lengths = wave_lengths
        self.wave_nos = 2. * torch.pi / wave_lengths
        self.discretization_size = discretization_size
        H = self._setup_H()
        self.register_buffer('H', H, persistent=False)

    def _setup_H(self):
        """
        set up the transfer function H for a Fresnel propagator.
        :return: the complex-valued transfer function H.
        """
        _, _, M_orig, N_orig = self.input_shape
        # zero padding.
        Mpad = M_orig // 4
        Npad = N_orig // 4
        M = M_orig + 2 * Mpad
        N = N_orig + 2 * Npad
        self.Npad, self.Mpad = Npad, Mpad

        xx, yy = get_coordinate(M, N, 1, 1)
        # Spatial frequency
        fx = xx / (self.discretization_size * N)  # max frequency = 1/(2*pixel_size)
        fy = yy / (self.discretization_size * M)

        fx = torch.fft.ifftshift(fx)
        fy = torch.fft.ifftshift(fy)

        squared_sum = (fx ** 2 + fy ** 2)[None][None]
        phi = - torch.pi * self.distance * self.wave_lengths.view(1, 3, 1, 1) * squared_sum
        H = torch.exp(1j * phi)
        return H

    def forward(self, input_field):
        Npad, Mpad = self.Npad, self.Mpad
        padded_input_field = F.pad(input=input_field, pad=[Npad, Npad, Mpad, Mpad])

        objFT = torch.fft.fft2(padded_input_field)
        out_field = torch.fft.ifft2(objFT * self.H)
        return out_field[:, :, Mpad:-Mpad, Npad:-Npad]


def get_one_phase_shift_thickness(wave_lengths, refractive_index):
    """Calculate the thickness (in meter) of a phaseshift of 2pi.
    """
    # refractive index difference
    delta_N = refractive_index - 1.
    # wave number
    wave_nos = 2. * torch.pi / wave_lengths
    two_pi_thickness = (2. * torch.pi) / (wave_nos * delta_N)
    return two_pi_thickness
