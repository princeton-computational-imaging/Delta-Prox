import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torch
from torchlight.data import SingleImageDataset

from dprox import *
from dprox import Variable
from dprox.linop.conv import conv_doe
from dprox.utils import *

from .doe_model import RGBCollimator, center_crop, img_psf_conv

circular = True
aperture_diameter = 3e-3
sensor_distance = 15e-3  # Distance of sensor to aperture
refractive_idcs = torch.tensor([1.4648, 1.4599, 1.4568])  # Refractive idcs of the phaseplate
wave_lengths = torch.tensor([460, 550, 640]) * 1e-9  # Wave lengths to be modeled and optimized for
num_steps = 10001  # Number of SGD steps
# patch_size = 1248  # Size of patches to be extracted from images, and resolution of simulated sensor
patch_size = 748  # Size of patches to be extracted from images, and resolution of simulated sensor
sample_interval = 2e-6  # Sampling interval (size of one "pixel" in the simulated wavefront)
wave_resolution = 1496, 1496  # Resolution of the simulated wavefront


device = torch.device('cuda')


def normalize_psf(psf, range=1, mode='band'):
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
    # Get the dimensions of the array
    height, width = arr.shape[:2]

    # Calculate the indices for the center 100x100 region
    start_row = int((height - size) / 2)
    end_row = start_row + size
    start_col = int((width - size) / 2)
    end_col = start_col + size

    # Crop the array to the center 100x100 region
    cropped_arr = arr[start_row:end_row, start_col:end_col]

    # Return the cropped array
    return cropped_arr

def build_doe_model():
    rgb_collim_model = RGBCollimator(sensor_distance,
                                     refractive_idcs=refractive_idcs,
                                     wave_lengths=wave_lengths,
                                     patch_size=patch_size,
                                     sample_interval=sample_interval,
                                     wave_resolution=wave_resolution,
                                     ).to(device)
    return rgb_collim_model


def build_baseline_profile(rgb_collim_model):
    k = 2 * torch.pi / wave_lengths[1]
    fresnel_phase = - k * ((rgb_collim_model.xx**2 + rgb_collim_model.yy**2)[None][None] / (2 * sensor_distance))
    fresnel_phase = fresnel_phase % (torch.pi * 2)
    height_map = rgb_collim_model.height_map.phase_to_height_map(fresnel_phase, 1)
    fresnel_phase_c = rgb_collim_model.height_map.get_phase_profile(height_map)
    return fresnel_phase_c


def load_sample_img(path='./8068.jpg', keep_ratio=True, patch_size=736):
    img = Image.open(path)
    ps = patch_size
    if keep_ratio:
        ps = min(img.height,img.width)
    img = center_crop(img, ps, ps)
    img = img.resize((patch_size, patch_size), Image.BICUBIC)
    x = torch.from_numpy(np.array(img).transpose((2, 0, 1)) / 255.)[None].to(device)
    return x


def sanity_check(psf):
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
   