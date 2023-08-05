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


def img_psf_conv(img, psf, circular=False):
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


class HeightMap(nn.Module):
    def __init__(self, height_map_shape, wave_lengths, refractive_idcs, xx, yy, sensor_distance):
        super().__init__()
        self.wave_lengths = wave_lengths
        self.refractive_idcs = refractive_idcs
        delta_N = refractive_idcs.view([1, -1, 1, 1]) - 1.
        # wave number
        wave_nos = 2. * torch.pi / wave_lengths
        wave_nos = wave_nos.view([1, -1, 1, 1])
        self.register_buffer('delta_N', delta_N, persistent=False)
        self.register_buffer('wave_nos', wave_nos, persistent=False)
        self.xx = xx
        self.yy = yy
        self.sensor_distance = sensor_distance

        # height_map_sqrt = torch.ones(size=height_map_shape) * 1e-4
        height_map_sqrt = self._fresnel_phase_init(1)
        self.height_map_sqrt = nn.Parameter(height_map_sqrt)

    def _fresnel_phase_init(self, idx=1):
        """
        calculate the Fresnel lens phase and convert it to a height map.

        :param idx: idx is an optional parameter that specifies the index of the wavelength to use in
        the calculation. It is set to 1 by default, defaults to 1 (optional)
        :return: the square root of the height map calculated from the Fresnel phase.
        """
        k = 2 * torch.pi / self.wave_lengths[idx]
        fresnel_phase = - k * ((self.xx**2 + self.yy**2)[None][None] / (2 * self.sensor_distance))
        fresnel_phase = fresnel_phase % (torch.pi * 2)
        height_map = self.phase_to_height_map(fresnel_phase, idx)
        return height_map ** 0.5

    def get_phase_profile(self, height_map=None):
        """
        calculate the phase profile of a height map using wave numbers and phase delay.

        :param height_map: A 2D tensor representing the height map of the surface. It is used to
        calculate the phase delay induced by the height field
        :return: a complex exponential phase profile calculated from the input height map.
        """
        if height_map is None:
            height_map = torch.square(self.height_map_sqrt + 1e-7)
        # phase delay indiced by height field
        phi = self.wave_nos * self.delta_N * height_map
        return torch.exp(1j * phi)

    def phase_to_height_map(self, phi, wave_length_idx=1):
        """
        take in a phase map and return a corresponding height map using the given wave
        length and refractive index.

        :param phi: The phase profile of the height map at a specific wavelength associated with `wave_length_idx`
        :param wave_length_idx: The index of the wavelength in the list of available wavelengths. This
        is used to retrieve the corresponding `delta_N` value for the given wavelength, defaults to 1
        (optional)
        :return: a height map calculated from the input phase and other parameters such as wave length
        and delta N.
        """
        wave_length = self.wave_lengths[wave_length_idx]
        delta_n = self.delta_N.view(-1)[wave_length_idx]
        k = 2. * torch.pi / wave_length
        phi = phi % (2 * torch.pi)
        height_map = phi / k / delta_n
        return height_map


class RGBCollimator(nn.Module):
    def __init__(self,
                 sensor_distance,
                 refractive_idcs,
                 wave_lengths,
                 patch_size,
                 sample_interval,
                 wave_resolution,
                 ):
        super().__init__()
        self.wave_res = wave_resolution
        self.wave_lengths = wave_lengths
        self.sensor_distance = sensor_distance
        self.sample_interval = sample_interval
        self.patch_size = patch_size
        self.refractive_idcs = refractive_idcs
        self._init_setup()

    def get_psf(self, phase_profile=None):
        """
        calculate the point spread function (PSF) of an optical system given a phase
        profile and other parameters.

        :param phase_profile: A 2D tensor representing the phase profile of the optical system. It is
        multiplied element-wise with the input field before propagation
        :return: a PSF (Point Spread Function) which is a 2D tensor representing the intensity
        distribution of the image formed by a point source after passing through the optical system.
        """
        if phase_profile is None:
            phase_profile = self.height_map.get_phase_profile()
        field = phase_profile * self.input_field
        field = self.aperture * field
        field = self.propagator(field)
        psfs = (torch.abs(field) ** 2).float()
        psfs = area_downsampling(psfs, self.patch_size)
        # psfs = psfs / psfs.sum(dim=[2, 3], keepdim=True)
        psfs = psfs / psfs.sum()
        return psfs

    def forward(self, input_img, phase_profile=None, circular=False):
        psfs = self.get_psf(phase_profile)
        output_image = img_psf_conv(input_img, psfs, circular=circular)
        return output_image, psfs

    def _init_setup(self):
        input_field = torch.ones((1, len(self.wave_lengths), self.wave_res[0], self.wave_res[1]))
        self.register_buffer("input_field", input_field, persistent=False)

        xx, yy = get_coordinate(self.wave_res[0], self.wave_res[1],
                                self.sample_interval, self.sample_interval)
        self.register_buffer("xx", xx, persistent=False)
        self.register_buffer("yy", yy, persistent=False)

        aperture = self._get_circular_aperture(xx, yy)
        self.register_buffer("aperture", aperture, persistent=False)

        self.height_map = self._get_height_map()
        self.propagator = self._get_propagator()

    def _get_height_map(self):
        height_map_shape = (1, 3, self.wave_res[0], self.wave_res[1])
        height_map = HeightMap(height_map_shape,
                               self.wave_lengths,
                               self.refractive_idcs,
                               self.xx, self.yy,
                               self.sensor_distance)
        return height_map

    def _get_propagator(self):
        input_shape = (1, 3, self.wave_res[0], self.wave_res[1])
        propagator = FresnelPropagator(input_shape,
                                       self.sensor_distance,
                                       self.sample_interval,
                                       self.wave_lengths)
        return propagator

    def _get_circular_aperture(self, xx, yy):
        max_val = xx.max()
        r = torch.sqrt(xx ** 2 + yy ** 2)
        aperture = (r < max_val).float()[None][None]
        return aperture


@dataclass
class DOEModelConfig:
    circular: bool = True  # circular convolution
    aperture_diameter: float = 3e-3  # aperture diameter
    sensor_distance: float = 15e-3  # Distance of sensor to aperture
    refractive_idcs = torch.tensor([1.4648, 1.4599, 1.4568])  # Refractive idcs of the phaseplate
    wave_lengths = torch.tensor([460, 550, 640]) * 1e-9  # Wave lengths to be modeled and optimized for
    num_steps: int = 10001  # Number of SGD steps
    # patch_size = 1248  # Size of patches to be extracted from images, and resolution of simulated sensor
    patch_size: int = 748  # Size of patches to be extracted from images, and resolution of simulated sensor
    sample_interval: float = 2e-6  # Sampling interval (size of one "pixel" in the simulated wavefront)
    wave_resolution = 1496, 1496  # Resolution of the simulated wavefront
    model_kwargs: dict = field(default_factory=dict)


def build_doe_model(config: DOEModelConfig = DOEModelConfig()):
    """
    build a DOE (Diffractive Optical Element) model using the input configuration.

    :param config: DOEModelConfig object that contains the following parameters:
    :type config: DOEModelConfig
    :return: The function `build_doe_model` is returning an instance of the `RGBCollimator` class, which
    is initialized with the parameters passed in the `DOEModelConfig` object `config`.
    """
    rgb_collim_model = RGBCollimator(config.sensor_distance,
                                     refractive_idcs=config.refractive_idcs,
                                     wave_lengths=config.wave_lengths,
                                     patch_size=config.patch_size,
                                     sample_interval=config.sample_interval,
                                     wave_resolution=config.wave_resolution,
                                     )
    return rgb_collim_model


def build_baseline_profile(rgb_collim_model: RGBCollimator):
    """
    build a baseline profile for a given RGB collimator model by calculating the Fresnel
    phase and height map.

    :param rgb_collim_model: An instance of the RGBCollimator class, which likely represents a system
    for collimating light of different wavelengths (red, green, and blue) onto a sensor or detector
    :type rgb_collim_model: RGBCollimator
    :return: the fresnel phase profile of the collimator model after applying the fresnel phase shift
    due to propagation to a sensor distance. The fresnel phase profile is calculated using the height
    map obtained from the phase-to-height map conversion function.
    """
    k = 2 * torch.pi / rgb_collim_model.wave_lengths[1]
    fresnel_phase = - k * ((rgb_collim_model.xx**2 + rgb_collim_model.yy**2)[None][None] / (2 * rgb_collim_model.sensor_distance))
    fresnel_phase = fresnel_phase % (torch.pi * 2)
    height_map = rgb_collim_model.height_map.phase_to_height_map(fresnel_phase, 1)
    fresnel_phase_c = rgb_collim_model.height_map.get_phase_profile(height_map)
    return fresnel_phase_c
