import torch
from .common import *


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

    def get_phase_profile(self, height_map=None, refractive_len=None):
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
        if refractive_len is not None:
            phi += refractive_len.to(phi.device)
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
                 aperture_type='half_circular'
                 ):
        super().__init__()
        self.wave_res = wave_resolution
        self.wave_lengths = wave_lengths
        self.sensor_distance = sensor_distance
        self.sample_interval = sample_interval
        self.patch_size = patch_size
        self.refractive_idcs = refractive_idcs
        self._init_setup(aperture_type=aperture_type)

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
            phase_profile = self.height_map.get_phase_profile(refractive_len=self.refractive_len)
        field = phase_profile * self.input_field
        field = self.aperture * field
        field = self.propagator(field)
        psfs = (torch.abs(field) ** 2).float()
        psfs = area_downsampling(psfs, self.patch_size)
        psfs = psfs / psfs.sum(dim=[2, 3], keepdim=True)
        # psfs = psfs / psfs.sum()
        return psfs

    def forward(self, input_img, phase_profile=None, circular=False):
        psfs = self.get_psf(phase_profile)
        output_image = img_psf_conv(input_img, psfs, circular=circular)
        return output_image, psfs

    def _init_setup(self, aperture_type):
        input_field = torch.ones((1, len(self.wave_lengths), self.wave_res[0], self.wave_res[1]))
        self.register_buffer("input_field", input_field, persistent=False)

        xx, yy = get_coordinate(self.wave_res[0], self.wave_res[1],
                                self.sample_interval, self.sample_interval)
        self.register_buffer("xx", xx, persistent=False)
        self.register_buffer("yy", yy, persistent=False)

        if aperture_type == 'half_circular':
            aperture = self._get_halfcircular_aperture(xx, yy)
        elif aperture_type == 'circular':
            aperture = self._get_circular_aperture(xx, yy)
        else:
            raise ImportError(f"Aperture type {aperture_type} not supported")
        self.register_buffer("aperture", aperture, persistent=False)

        self.height_map = self._get_height_map()
        self.propagator = self._get_propagator()

        self.refractive_len = self._get_refractive_len()

    def _get_height_map(self):
        height_map_shape = (1, 1, self.wave_res[0], self.wave_res[1])
        height_map = HeightMap(height_map_shape,
                               self.wave_lengths,
                               self.refractive_idcs,
                               self.xx, self.yy,
                               self.sensor_distance)
        return height_map

    def _get_propagator(self):
        input_shape = (1, len(self.wave_lengths), self.wave_res[0], self.wave_res[1])
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

    def _get_halfcircular_aperture(self, xx, yy):
        max_val = xx.max()
        r = torch.sqrt(xx ** 2 + yy ** 2)
        aperture = ((yy > 0) * (r < max_val)).float()[None][None]
        return aperture
    
    def _get_refractive_len(self):
        refractive_len = []
        for idx in range(len(self.wave_lengths)):   
            k = 2 * torch.pi / self.wave_lengths[idx]
            fresnel_phase = - k * ((self.xx**2 + self.yy**2)[None][None] / (2 * self.sensor_distance))
            fresnel_phase = fresnel_phase % (torch.pi * 2)
            refractive_len.append(fresnel_phase)
        return torch.cat(refractive_len, dim = 1)


wvls = torch.tensor([460, 550, 640])
wvl_um = wvls * 1e-3
RI = (1 + 0.6961663 / (1 - (0.0684043 / wvl_um) ** 2) + 0.4079426 / (1 - (0.1162414 / wvl_um) ** 2) + 0.8974794 / (1 - (9.896161 / wvl_um) ** 2)) ** .5 

@dataclass
class DOEModelConfig:
    circular: bool = True  # circular convolution
    aperture_diameter: float = 9e-3  # aperture diameter
    aperture_type: str = 'half_circular'  # Type of aperture to be used
    sensor_distance: float = 50e-3  # Distance of sensor to aperture
    wave_lengths = wvls * 1e-9  # Wave lengths to be modeled and optimized for
    refractive_idcs = RI # Refractive idcs of the phaseplate
    num_steps: int = 10001  # Number of SGD steps
    patch_size: int = 512  # Size of patches to be extracted from images, and resolution of simulated sensor
    sample_interval: float = 5.4e-6  # Sampling interval (size of one "pixel" in the simulated wavefront)
    wave_resolution = 1536, 1536  # Resolution of the simulated wavefront
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
                                     aperture_type=config.aperture_type
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
