#%%
from doe_model import RGBCollimator, center_crop, img_psf_conv
import torch
import numpy as np


def main():
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

    rgb_collim_model = RGBCollimator(sensor_distance,
                                     refractive_idcs=refractive_idcs,
                                     wave_lengths=wave_lengths,
                                     patch_size=patch_size,
                                     sample_interval=sample_interval,
                                     wave_resolution=wave_resolution,
                                     ).to(device)

    k = 2 * torch.pi / wave_lengths[1]
    fresnel_phase = - k * ((rgb_collim_model.xx**2 + rgb_collim_model.yy**2)[None][None] / (2 * sensor_distance))
    fresnel_phase = fresnel_phase % (torch.pi * 2)

    psf = rgb_collim_model.get_psf(fresnel_phase)

    import PIL.Image as Image
    import matplotlib.pylab as plt
    img = Image.open('./8068.jpg')
    img = center_crop(img, patch_size // 2, patch_size // 2)
    img = img.resize((patch_size, patch_size), Image.BICUBIC)
    x = torch.from_numpy(np.array(img).transpose((2, 0, 1)) / 255.)[None].to(device)

    output_image = img_psf_conv(x, psf, circular=False)
    print(psf.shape)
    print(output_image.shape)
    plt.imshow(output_image[0].permute(1,2,0).cpu().numpy())
    plt.show()
    
    from dprox.linop.conv import conv_doe
    from dprox import Variable, CompGraph
    conv = conv_doe(Variable(), psf)
    out = conv.forward([x])[0]
    plt.imshow(out[0].permute(1,2,0).cpu().numpy())
    
    graph = CompGraph(conv)
    graph.sanity_check()
    print(torch.allclose(out, output_image.float()))
    

    
main()
# %%
