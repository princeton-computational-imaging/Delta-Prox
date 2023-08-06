import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torch
from dprox import Variable
from dprox.linop.conv import conv_doe
from dprox.utils.misc import outlier_correct
from torchlight.data import SingleImageDataset

from .doe_model import img_psf_conv


def load_sample_img(path, keep_ratio=True, patch_size=748):
    img = Image.open(path)
    ps = patch_size
    if keep_ratio:
        ps = min(img.height, img.width)
    img = center_crop(img, ps, ps)
    img = img.resize((patch_size, patch_size), Image.BICUBIC)
    x = torch.from_numpy(np.array(img).transpose((2, 0, 1)) / 255.)[None]
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


def center_crop(im, new_height, new_width):
    width, height = im.size   # Get dimensions

    left = round((width - new_width) / 2)
    top = round((height - new_height) / 2)
    x_right = round(width - new_width) - left
    x_bottom = round(height - new_height) - top
    right = width - x_right
    bottom = height - x_bottom

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def normalize_psf2(psf, range=1, mode='band'):
    def norm(psf):
        if mode == 'band':
            psf[:, :, 0] = (psf[:, :, 0] - psf[:, :, 0].min()) / (psf[:, :, 0].max() - psf[:, :, 0].min())
            psf[:, :, 1] = (psf[:, :, 1] - psf[:, :, 1].min()) / (psf[:, :, 1].max() - psf[:, :, 1].min())
            psf[:, :, 2] = (psf[:, :, 2] - psf[:, :, 2].min()) / (psf[:, :, 2].max() - psf[:, :, 2].min())
        else:
            psf = (psf - psf.min()) / (psf.max() - psf.min())
        return psf

    psf = norm(psf)
    psf = psf.clip(0, range)
    psf = norm(psf)
    return psf


def normalize_psf(psf: np.ndarray, clip_percentile=0.01, bandwise=False):
    """
    normalize a point spread function (PSF) by dividing it by its sum, correcting for
    outliers, and normalizing the maximum value to 1 for visualization.

    :param psf: a numpy array representing the point spread function (PSF) of an imaging system
    :type psf: np.ndarray
    :param clip_percentile: represent the percentage of data that is considered as outliers. In this
    function, it is used to calculate the lower and upper percentiles of the data that are not
    considered as outliers. 
    :param bandwise: A boolean parameter that specifies whether to normalize the PSF bandwise or not. If
    set to True, the PSF will be normalized separately for each band. If set to False, the PSF will be
    normalized across all bands, defaults to False (optional)
    :return: a normalized and outlier-corrected version of the input PSF (point spread function) array.
    If `bandwise` is True, the normalization is done separately for each band of the PSF. The
    normalization involves dividing the PSF by its sum (or sum along each band if `bandwise` is True),
    then correcting for outliers using the `outlier_correct`
    """
    if bandwise:
        psf = psf / psf.sum(axis=(0, 1), keepdims=True)
    else:
        psf = psf / psf.sum()  # sum to 1
    psf = outlier_correct(psf, p=clip_percentile)
    psf = psf / psf.max()  # normalize the max value to 1 for visualization
    return psf


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
    data = data[200:400, 200:400]
    H, W = data.shape
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


class Dataset(SingleImageDataset):
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        img = img.transpose(1, 2, 0)
        img = cv2.resize(img, dsize=(768, 768),
                         interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        return img
