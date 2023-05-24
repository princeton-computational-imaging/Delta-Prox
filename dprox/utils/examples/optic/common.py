import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torch
from dataclasses import dataclass, field
from torchlight.data import SingleImageDataset

from dprox import *
from dprox import Variable
from dprox.linop.conv import conv_doe
from dprox.utils import *

from .doe_model import RGBCollimator, center_crop, img_psf_conv


device = torch.device('cuda')


@dataclass
class DOEModelConfig:
    circular: bool = True # circular convolution
    aperture_diameter: float = 3e-3 # aperture diameter
    sensor_distance: float = 15e-3  # Distance of sensor to aperture
    refractive_idcs = torch.tensor([1.4648, 1.4599, 1.4568])  # Refractive idcs of the phaseplate
    wave_lengths = torch.tensor([460, 550, 640]) * 1e-9  # Wave lengths to be modeled and optimized for
    num_steps: int = 10001  # Number of SGD steps
    # patch_size = 1248  # Size of patches to be extracted from images, and resolution of simulated sensor
    patch_size: int = 748  # Size of patches to be extracted from images, and resolution of simulated sensor
    sample_interval: float = 2e-6  # Sampling interval (size of one "pixel" in the simulated wavefront)
    wave_resolution = 1496, 1496  # Resolution of the simulated wavefront
    model_kwargs: dict = field(default_factory=dict)


def normalize_psf(psf, range=1, mode='band'):
    """
    normalize a given PSF (point spread function) image by scaling its pixel values to a
    specified range and mode.
    
    :param psf: a 3-dimensional numpy array representing a point spread function (PSF) image
    :param range: The maximum value that the PSF can take after normalization. Any values above this
    range will be clipped to this maximum value, defaults to 1 (optional)
    :param mode: The mode parameter determines how the PSF (point spread function) is normalized. If
    mode is set to 'band', each color band of the PSF is normalized separately. If mode is set to
    anything else, the entire PSF is normalized as a single entity, defaults to band (optional)
    :return: the normalized PSF (point spread function) with values clipped between 0 and the specified
    range.
    """
    def norm(psf):
        if mode == 'band':
            psf[:,:,0] = (psf[:,:,0]-psf[:,:,0].min()) / (psf[:,:,0].max()-psf[:,:,0].min())
            psf[:,:,1] = (psf[:,:,1]-psf[:,:,1].min()) / (psf[:,:,1].max()-psf[:,:,1].min())
            psf[:,:,2] = (psf[:,:,2]-psf[:,:,2].min()) / (psf[:,:,2].max()-psf[:,:,2].min())
        else:
            psf = (psf - psf.min()) / (psf.max() - psf.min())
        return psf
    
    psf = norm(psf)
    psf = psf.clip(0, range)
    psf = norm(psf)
    return psf


def crop_center_region(arr, size=150):
    """
    crop a center region of a given size from a 2D array.
    
    :param arr: a numpy array representing an image
    :param size: The size of the square region to be cropped from the center of the input array. The
    default value is 150, defaults to 150 (optional)
    :return: a cropped version of the input array, where the center region of the array is extracted
    based on the specified size parameter.
    """
    # Get the dimensions of the array
    height, width = arr.shape[:2]

    # Calculate the indices for the center sizexsize region
    start_row = int((height - size) / 2)
    end_row = start_row + size
    start_col = int((width - size) / 2)
    end_col = start_col + size

    # Crop the array to the center sizexsize region
    cropped_arr = arr[start_row:end_row, start_col:end_col]

    # Return the cropped array
    return cropped_arr


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
                                     ).to(device)
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


def load_sample_img(path='./8068.jpg', keep_ratio=True, patch_size=736):
    img = Image.open(path)
    ps = patch_size
    if keep_ratio:
        ps = min(img.height, img.width)
    img = center_crop(img, ps, ps)
    img = img.resize((patch_size, patch_size), Image.BICUBIC)
    x = torch.from_numpy(np.array(img).transpose((2, 0, 1)) / 255.)[None].to(device)
    return x


def sanity_check(psf, circular=True):
    """
    perform a sanity check on the output of `img_psf_conv` and `conv_doe` functions.
    
    :param psf: The point spread function (PSF) is a mathematical function that describes how an imaging
    system blurs a point source of light. It is often used in image processing and analysis to model the
    effects of image blurring and to deconvolve images
    :param circular: A boolean parameter that determines whether the PSF should be treated as circular
    or not during convolution. If set to True, the PSF will be wrapped around the edges of the image to
    simulate a circular convolution. If set to False, the PSF will be treated as a non-circular
    convolution and, defaults to True (optional)
    :return: a tuple containing two elements: the input image x and the output image out, which is the
    result of convolving x with the given point spread function (psf) using the img_psf_conv function
    and the conv_doe operator. The function also performs a sanity check to verify if the output image
    is close enough to the expected result.
    """
    x = load_sample_img()

    output_image = img_psf_conv(x, psf, circular=circular)
    print(psf.shape)
    print(output_image.shape)

    op = conv_doe(Variable(), psf, circular=circular)
    out = op.forward([x])[0]
    ad = op.adjoint([out])[0]
    print('Check passed ?', torch.allclose(out, output_image.float()))

    return x, out


class Dataset(SingleImageDataset):
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        img = img.transpose(1, 2, 0)
        img = cv2.resize(img, dsize=(752, 752),
                         interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        return img


def subplot(data, path):
    _, axes = plt.subplots(1, 3)
    for i in range(3):
        img = data[0, i].detach().cpu().numpy()
        im = axes[i].imshow(img)
        plt.colorbar(im, ax=axes[i])
    plt.savefig(path)
    plt.close()


def plot(data, path):
    plt.figure()
    data = data.detach().squeeze().cpu().numpy()
    # data = (data-data.min())/(data.max()-data.min())
    plt.imshow(data)
    plt.colorbar()
    plt.savefig(path)
    plt.close()


def plot3d(data, path):
    data = data.detach().squeeze().cpu().numpy()
    data = data[200:400,200:400]
    H,W = data.shape
    # x = np.floor(np.linspace(0, H, 50)).astype('int') - 1
    # x = np.floor(np.linspace(0, H, 50)).astype('int') - 1
    x = np.linspace(0, H, 50)
    y = np.linspace(0, W, 50)

    X, Y = np.meshgrid(x, y)
    # Z = data[X,Y]
    Z = data

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D contour2')
    plt.show()
    plt.savefig(path)
    plt.close()
   